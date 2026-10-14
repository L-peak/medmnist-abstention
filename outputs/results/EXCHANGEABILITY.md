# Step A control: exchangeable calibration/evaluation halves (generated)

10 random 50/50 splits of the **test** set per subset, each used in both directions, so the calibration and evaluation halves are exchangeable by construction. The threshold (and the conformal quantile) is selected on the calibration half only, exactly as in the manuscript's deployable protocol.

## Deployable protocol inside the test split

| dataset        |   mean_risk |   median_risk |   max_risk |   mean_risk_gap |   over_budget_rate |
|:---------------|------------:|--------------:|-----------:|----------------:|-------------------:|
| bloodmnist     |       0.044 |         0.044 |      0.051 |          -0     |               0    |
| breastmnist    |       0.101 |         0.089 |      0.214 |           0.007 |               0.45 |
| chestmnist     |       0.058 |         0.058 |      0.059 |          -0     |               0    |
| dermamnist     |       0.1   |         0.099 |      0.136 |           0.001 |               0.48 |
| octmnist       |       0.104 |         0.1   |      0.186 |           0.005 |               0.49 |
| organamnist    |       0.068 |         0.068 |      0.073 |          -0     |               0    |
| organcmnist    |       0.084 |         0.084 |      0.092 |          -0     |               0    |
| organsmnist    |       0.1   |         0.1   |      0.124 |          -0     |               0.51 |
| pathmnist      |       0.1   |         0.1   |      0.119 |           0     |               0.49 |
| pneumoniamnist |       0.103 |         0.101 |      0.186 |           0.004 |               0.5  |
| retinamnist    |       0.187 |         0.172 |      0.5   |           0.128 |               0.7  |
| tissuemnist    |       0.1   |         0.1   |      0.107 |           0     |               0.52 |

- mean evaluation-half risk over all folds and scores: **0.097** (budget 0.1)
- mean calibration-half risk at the selected thresholds: 0.084
- mean risk gap (evaluation - calibration): **+0.012**, mean absolute gap 0.022, sd 0.060
- folds above the budget: 0.35 (expected: the guarantee is marginal, so finite-half noise puts part of the folds above the line even when the protocol is valid)

## Split-conformal coverage inside the test split

| dataset        |   conformal_aps |   conformal_lac |
|:---------------|----------------:|----------------:|
| bloodmnist     |           0.999 |           0.9   |
| breastmnist    |           0.987 |           0.915 |
| dermamnist     |           0.954 |           0.901 |
| octmnist       |           0.954 |           0.899 |
| organamnist    |           1     |           0.9   |
| organcmnist    |           0.998 |           0.9   |
| organsmnist    |           0.99  |           0.9   |
| pathmnist      |           0.982 |           0.9   |
| pneumoniamnist |           0.959 |           0.897 |
| retinamnist    |           0.905 |           0.901 |
| tissuemnist    |           0.962 |           0.9   |

- conformal_lac: mean evaluation-half coverage **0.901** (nominal >= 0.90); below nominal in 0.47 of folds
- conformal_aps: mean evaluation-half coverage **0.972** (nominal >= 0.90); below nominal in 0.05 of folds

## Validation -> test (manuscript Table 5) vs exchangeable halves

| dataset        |   validation_to_test_mean_risk |   exchangeable_halves_mean_risk |   ratio |
|:---------------|-------------------------------:|--------------------------------:|--------:|
| bloodmnist     |                          0.044 |                           0.044 |    1    |
| breastmnist    |                          0.143 |                           0.101 |    1.42 |
| chestmnist     |                          0.066 |                           0.058 |    1.14 |
| dermamnist     |                          0.111 |                           0.1   |    1.11 |
| octmnist       |                          0.251 |                           0.104 |    2.41 |
| organamnist    |                          0.067 |                           0.068 |    0.99 |
| organcmnist    |                          0.084 |                           0.084 |    1    |
| organsmnist    |                          0.213 |                           0.1   |    2.13 |
| pathmnist      |                          0.124 |                           0.1   |    1.24 |
| pneumoniamnist |                          0.138 |                           0.103 |    1.34 |
| retinamnist    |                          0.208 |                           0.181 |    1.15 |
| tissuemnist    |                          0.1   |                           0.1   |    1    |

- validation -> test: mean risk 0.129 over the same score set
- exchangeable halves: mean risk 0.095
