"""Isolated federated client evaluation and local training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from .blindspot import ClassStatistics, blindspot_class_weights
from .fedprox import proximal_penalty


@dataclass
class ClientUpdate:
    """All aggregates uploaded by a client; never contains raw examples."""

    client_id: int
    delta: dict[str, Tensor]
    train_sample_count: int
    class_train_counts: NDArray[np.int64]
    validation_statistics: ClassStatistics
    train_loss: float


class FederatedClient:
    """Client wrapper holding only its local dataset index sets."""

    def __init__(
        self,
        client_id: int,
        dataset: Any,
        targets: NDArray[np.int64],
        train_indices: Sequence[int],
        validation_indices: Sequence[int],
        *,
        num_classes: int,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        self.client_id = int(client_id)
        self.dataset = dataset
        self.targets = np.asarray(targets, dtype=np.int64)
        self.train_indices = [int(index) for index in train_indices]
        self.validation_indices = [int(index) for index in validation_indices]
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        if not self.train_indices or not self.validation_indices:
            raise ValueError(f"Client {client_id} has an empty local split.")
        combined = self.train_indices + self.validation_indices
        if min(combined) < 0 or max(combined) >= len(dataset):
            raise IndexError(f"Client {client_id} has out-of-range dataset indexes.")

    def _loader(
        self, indices: Sequence[int], *, shuffle: bool, seed: int
    ) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(seed)
        drop_last = (
            shuffle
            and len(indices) > 1
            and len(indices) % self.batch_size == 1
        )
        return DataLoader(
            Subset(self.dataset, list(indices)),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=generator,
            drop_last=drop_last,
        )

    @torch.no_grad()
    def validation_statistics(
        self, model: nn.Module, *, device: torch.device, seed: int
    ) -> ClassStatistics:
        """Evaluate the broadcast global model before any local update."""

        model.eval()
        count = np.zeros(self.num_classes, dtype=np.int64)
        loss_sum = np.zeros(self.num_classes, dtype=np.float64)
        correct = np.zeros(self.num_classes, dtype=np.int64)
        criterion = nn.CrossEntropyLoss(reduction="none")
        for features, labels in self._loader(
            self.validation_indices, shuffle=False, seed=seed
        ):
            features = features.to(device, non_blocking=self.pin_memory)
            labels = labels.to(device, non_blocking=self.pin_memory)
            logits = model(features)
            losses = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            labels_cpu = labels.detach().cpu().numpy()
            losses_cpu = losses.detach().cpu().numpy()
            correct_cpu = predictions.eq(labels).detach().cpu().numpy()
            for class_id in range(self.num_classes):
                mask = labels_cpu == class_id
                count[class_id] += int(mask.sum())
                loss_sum[class_id] += float(losses_cpu[mask].sum())
                correct[class_id] += int(correct_cpu[mask].sum())
        return ClassStatistics(count=count, loss_sum=loss_sum, correct=correct)

    def train(
        self,
        global_model: nn.Module,
        blindspot_weights: NDArray[np.floating],
        *,
        device: torch.device,
        optimizer_config: Mapping[str, Any],
        local_epochs: int,
        blindspot_strength: float,
        use_blindspot_loss: bool,
        use_fedprox: bool,
        fedprox_mu: float,
        seed: int,
    ) -> ClientUpdate:
        """Evaluate, clone, train, and return a detached CPU model delta."""

        local_model = copy.deepcopy(global_model).to(device)
        local_model.load_state_dict(global_model.state_dict(), strict=True)
        global_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in global_model.state_dict().items()
        }
        statistics = self.validation_statistics(
            local_model, device=device, seed=seed
        )
        global_parameters = {
            name: parameter.detach().to(device).clone()
            for name, parameter in global_model.named_parameters()
        }
        if use_blindspot_loss:
            weight_array = blindspot_class_weights(
                blindspot_weights, blindspot_strength
            )
            criterion = nn.CrossEntropyLoss(
                weight=torch.as_tensor(weight_array, device=device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        name = str(optimizer_config.get("name", "adam")).lower()
        kwargs = {
            "lr": float(optimizer_config["lr"]),
            "weight_decay": float(optimizer_config.get("weight_decay", 0.0)),
        }
        if name == "adam":
            optimizer = torch.optim.Adam(local_model.parameters(), **kwargs)
        elif name == "sgd":
            optimizer = torch.optim.SGD(
                local_model.parameters(),
                momentum=float(optimizer_config.get("momentum", 0.0)),
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported optimizer {name!r}.")

        local_model.train()
        loss_numerator = 0.0
        examples_seen = 0
        for epoch in range(int(local_epochs)):
            loader = self._loader(
                self.train_indices,
                shuffle=True,
                seed=seed + epoch,
            )
            for features, labels in loader:
                features = features.to(device, non_blocking=self.pin_memory)
                labels = labels.to(device, non_blocking=self.pin_memory)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(local_model(features), labels)
                if use_fedprox:
                    loss = loss + 0.5 * float(fedprox_mu) * proximal_penalty(
                        local_model, global_parameters
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Client {self.client_id} produced non-finite loss."
                    )
                loss.backward()
                optimizer.step()
                batch_size = int(labels.shape[0])
                loss_numerator += float(loss.detach().cpu()) * batch_size
                examples_seen += batch_size

        local_state = local_model.state_dict()
        if set(local_state) != set(global_state):
            raise RuntimeError("Local/global state dictionaries have different keys.")
        delta: dict[str, Tensor] = {}
        for key, global_tensor in global_state.items():
            local_tensor = local_state[key].detach().cpu()
            if local_tensor.shape != global_tensor.shape:
                raise RuntimeError(f"State shape changed for {key!r}.")
            if torch.is_floating_point(local_tensor):
                difference = local_tensor - global_tensor
                if not torch.all(torch.isfinite(difference)):
                    raise FloatingPointError(f"Non-finite client update for {key!r}.")
                delta[key] = difference
            else:
                delta[key] = torch.zeros_like(global_tensor)
        class_counts = np.bincount(
            self.targets[np.asarray(self.train_indices, dtype=np.int64)],
            minlength=self.num_classes,
        ).astype(np.int64)
        return ClientUpdate(
            client_id=self.client_id,
            delta=delta,
            train_sample_count=len(self.train_indices),
            class_train_counts=class_counts,
            validation_statistics=statistics,
            train_loss=loss_numerator / max(1, examples_seen),
        )

