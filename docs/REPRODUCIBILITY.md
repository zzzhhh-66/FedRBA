# Reproducibility protocol

## Frozen design

- Datasets: Fashion-MNIST, Default of Credit Card Clients, and Give Me Some
  Credit.
- Methods: FedAvg, FedAvg-ClassWeight, FedProx, FedRBA-ClassAgg, and FedRBA.
- Seeds: 1, 2, and 3.
- Main evaluation window: last 20 communication rounds.
- Main partition: label Dirichlet with the alpha values encoded in the matrix.
- Additional quantity-skew experiment: log-normal sigma 1.5.

## Split and leakage controls

1. A fixed stratified train/test split is created for each dataset and seed.
2. Tabular preprocessing is fitted only on the central training split.
3. The training split is partitioned across clients.
4. Every client reserves a local validation subset.
5. Global test data is not used to update blindspot weights or model parameters.
6. Operating-point thresholds and checkpoint choices are selected from
   validation predictions only.
7. The selected threshold is applied to test predictions once for reporting.

The operating-point tables include split hashes. For a given dataset, alpha,
and seed, all compared methods must have identical split hashes.

## Aggregation rules

For every run, metrics are first averaged across the last 20 rounds. The three
per-seed values are then summarized as:

```text
mean +/- sample standard deviation
```

The analysis additionally reports minority-collapse rate, defined by the
analysis scripts over the trailing window. Final-round-only comparisons and
test-selected best rounds are not used for the paper's main claims.

## Matrix completeness

Expected completed runs:

| Experiment | Runs |
|---|---:|
| Main matrix | 90 |
| Credit ablations | 18 |
| Explicit quantity skew | 15 |
| Validation-matched operating points | 45 |

The committed artifacts record complete matrices for all four groups.

## Tests

Run:

```bash
python -m pytest tests -q
```

The suite is offline and includes partition invariants, preprocessing leakage
checks, aggregation edge cases, matrix counts, operating-point selection, and
two-round CPU training for FedAvg and FedRBA.

## Archived environment

The original HPC runs used:

```text
Python: 3.10.0
PyTorch: 2.6.0+cu124
CUDA build: 12.4
```

The exact package requirements used by the runnable project are in
`requirements.txt`. Before a camera-ready release, also archive a fresh
`pip freeze`, the cluster job scripts, and the Git commit hash associated with
the final paper tables.

## Artifact mapping

- `artifacts/tables/main.csv`: complete multi-seed main table.
- `artifacts/tables/op.csv`: aggregated validation-matched operating points.
- `artifacts/tables/op_per_run.csv`: per-seed operating points and split hashes.
- `artifacts/tables/ablation.csv`: credit ablation summaries.
- `artifacts/tables/quantity.csv`: explicit quantity-skew summaries.
- `artifacts/audits/`: curve, partition, and source audits.
- `artifacts/FedRBA_Final_Paper_Results.xlsx`: formatted results workbook.
- `artifacts/figures/`: paper figures and captions.

