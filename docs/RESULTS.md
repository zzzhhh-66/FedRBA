# Verified results and claim boundary

## Evaluation convention

All main values are the mean of the last 20 communication rounds within each
run, followed by mean and standard deviation across seeds 1, 2, and 3. Credit
experiments emphasize PR-AUC, ROC-AUC, minority recall/F1, collapse rate, and
validation-matched operating points.

## Supported main finding

Relative to the strongest general-purpose baseline for minority recall in each
credit scenario, full FedRBA improves trailing-window minority recall by:

| Scenario | Recall improvement |
|---|---:|
| Default Credit, alpha = 0.1 | +17.91 percentage points |
| Default Credit, alpha = 0.3 | +8.88 percentage points |
| Give Me Some Credit, alpha = 0.3 | +5.91 percentage points |
| Default Credit with log-normal quantity skew | +5.78 percentage points |

The 10/20/30-round window sensitivity analysis preserves the direction of this
conclusion on all three main credit scenarios.

## Operating-point interpretation

At thresholds selected on validation data under a common rule, FedRBA is
generally comparable to the best baseline at fixed FPR and fixed recall, but is
not uniformly superior. Therefore the defensible claim is:

> FedRBA mitigates minority-class prediction collapse while maintaining
> competitive validation-matched operating-point performance.

The repository does not support a claim that FedRBA dominates every method on
all discrimination and operating-point metrics.

## Known trade-off

On Give Me Some Credit, FedRBA improves minority recall but sacrifices part of
the ranking performance relative to the strongest ROC-AUC/PR-AUC baselines.
The effect persists across 10/20/30-round windows and is not an artifact of
choosing the last 20 rounds.

Minority metrics also fluctuate substantially across communication rounds. For
that reason, the paper should report:

- three-seed mean and standard deviation;
- trailing-window summaries rather than only the final round;
- minority-collapse rate;
- a dedicated volatility/convergence figure;
- validation-matched operating points.

## Artifact provenance

The committed aggregate tables were recomputed from complete matrices:

- 90 main runs;
- 18 credit ablation runs;
- 15 explicit quantity-skew runs;
- 45 operating-point evaluations.

See `artifacts/audits/source_manifest.json` and the audit JSON files for source
counts and consistency checks. Raw per-round training outputs are intentionally
not committed.

