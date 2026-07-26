# Paper figure captions

## Figure 1 — Credit convergence

**Credit-risk convergence under label heterogeneity.** Curves show the three-seed mean and shaded regions show ±1 standard deviation across seeds. A centered five-round moving average is used only for visualization; all reported numerical results use unsmoothed per-round values and the registered final-20-round aggregation. FedRBA raises minority recall in all three credit scenarios, while PR-AUC can remain below the strongest ranking baseline.

## Figure 2 — Fashion-MNIST convergence

**Fashion-MNIST convergence across Non-IID severity.** Curves show the three-seed mean with ±1 standard-deviation bands. The full FedRBA method is competitive under moderate label skew (α=0.3), whereas FedProx is stronger under extreme skew (α=0.1). At α=1.0, the methods converge to similar aggregate performance.

## Figure 3 — Credit trade-off

**FedRBA trade-off relative to the strongest generic baseline.** Positive values favor FedRBA. For each scenario and metric, the generic comparison baseline is the best of FedAvg, class-weighted FedAvg, and FedProx according to the three-seed mean. FedRBA consistently improves minority recall and reduces collapse, but incurs ROC-AUC and PR-AUC costs, particularly on GMSC.

## Figure 4 — GMSC volatility

**Minority-recall volatility on GMSC.** The left panel shows raw and five-round-smoothed FedRBA trajectories for each seed; the right panel shows the three-seed mean and standard-deviation bands for all methods. The repeated collapse-and-recovery cycles demonstrate that the GMSC trade-off is a persistent client-sampling stability issue rather than a single final-round anomaly.

## Mandatory methods note

All figure smoothing is visual only. Table values, comparisons, collapse rates, and conclusions are calculated from unsmoothed curves. Fashion-MNIST uses 100 communication rounds and rounds 81–100 for final-window summaries; Default Credit and GMSC use 150 rounds and rounds 131–150.
