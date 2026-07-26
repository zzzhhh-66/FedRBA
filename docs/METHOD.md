# Method

## Problem setting

Let the server model be split into a feature backbone and a classifier:

```text
theta = (phi, W, b)
```

Each communication round samples a subset of clients. Every selected client
starts from the same broadcast model, trains locally, and returns a model delta,
its sample count, class counts, and aggregate per-class validation statistics.
Optimizer state is not shared between clients.

## Dynamic class blindspots

For class `c`, the server aggregates validation counts and correct predictions:

```text
error_c = 1 - correct_c / max(count_c, epsilon)
raw_q   = softmax(error / temperature)
q_t     = ema * q_(t-1) + (1 - ema) * raw_q
```

The normalized blindspot vector `q_t` assigns more mass to classes for which the
broadcast model performs poorly. The local class-loss weight is:

```text
weight_c = C * ((1 - strength) / C + strength * q_t,c)
```

where `C` is the number of classes. Weights have mean one.

## Reliability-aware aggregation

Backbone updates are weighted by client sample count and optional update
alignment. Classifier rows are aggregated independently. For client `i` and
class `c`:

```text
sufficiency_i,c = n_i,c / (n_i,c + tau)
alignment_i,c   = clipped cosine alignment with the class reference update
reliability_i,c = sufficiency_i,c * alignment_i,c^gamma
beta_i,c        proportional to n_i,c * reliability_i,c
```

Clients with no samples for a class receive exactly zero weight for that
classifier row. If no selected client has the class, that row is unchanged.
The server can also scale the aggregated classifier-row update using the
blindspot vector.

## Compared methods

| Name | Description |
|---|---|
| `fedavg` | Standard sample-count-weighted FedAvg |
| `fedavg_classweight` | FedAvg with dynamic blindspot-aware local class weights |
| `fedprox` | FedProx with the configured proximal coefficient |
| `fedrba_classagg` | Class-wise reliability aggregation without blindspot loss/scaling |
| `fedrba` | Full method |

## Credit ablations

The dedicated matrix builder evaluates:

- no blindspot-aware local loss;
- no class-wise reliability aggregation;
- no update-alignment term.

FedAvg and full FedRBA are reused from the main matrix to avoid redundant
training. The implementation switches are resolved in
`src/federated/fedrba.py`.

## Privacy boundary

FedRBA shares aggregate per-class validation statistics with the server, not
individual examples. This reduces data exposure compared with centralizing raw
records, but it is not a formal privacy mechanism. The implementation contains
neither differential privacy nor cryptographic secure aggregation.

