# MedMNIST abstention benchmark - summary

## Test accuracy (mean of seeds)

| dataset | seeds | test acc (mean) |
| --- | --- | --- |
| bloodmnist | 3 | 0.958 |
| breastmnist | 3 | 0.842 |
| chestmnist | 3 | 0.448 (subset; macro-AUC 0.656) |
| dermamnist | 3 | 0.751 |
| octmnist | 3 | 0.754 |
| organamnist | 3 | 0.934 |
| organcmnist | 3 | 0.913 |
| organsmnist | 3 | 0.776 |
| pathmnist | 3 | 0.898 |
| pneumoniamnist | 3 | 0.853 |
| retinamnist | 3 | 0.472 |
| tissuemnist | 3 | 0.668 |

Multi-label subsets report subset (exact-match) accuracy plus macro-AUC,
and their abstention risk is per-sample Hamming error.

## Best method per dataset (lowest AURC)

| dataset        | best_method   |   aurc |   e_aurc |   coverage_at_risk |
|:---------------|:--------------|-------:|---------:|-------------------:|
| bloodmnist     | mc_dropout    | 0.0044 |   0.0035 |             1      |
| breastmnist    | ensemble      | 0.0556 |   0.0397 |             0.8205 |
| chestmnist     | ensemble      | 0.0353 |   0.0187 |             1      |
| dermamnist     | ensemble      | 0.0808 |   0.0465 |             0.6633 |
| octmnist       | ensemble      | 0.0946 |   0.0598 |             0.593  |
| organamnist    | ensemble      | 0.0055 |   0.0031 |             1      |
| organcmnist    | ensemble      | 0.011  |   0.0074 |             1      |
| organsmnist    | ensemble      | 0.0668 |   0.0404 |             0.7151 |
| pathmnist      | ensemble      | 0.0339 |   0.0259 |             0.9415 |
| pneumoniamnist | temp_scaling  | 0.047  |   0.0365 |             0.8654 |
| retinamnist    | ensemble      | 0.2941 |   0.1319 |             0.15   |
| tissuemnist    | ensemble      | 0.145  |   0.0818 |             0.4063 |

## Does the winner change when using AUGRC instead of AURC?

| dataset        | best_by_AURC   | best_by_AUGRC   | agree   |   augrc_AURC_best |   augrc_AUGRC_best |
|:---------------|:---------------|:----------------|:--------|------------------:|-------------------:|
| bloodmnist     | mc_dropout     | ensemble        | False   |            0.0036 |             0.0032 |
| breastmnist    | ensemble       | ensemble        | True    |            0.0398 |             0.0398 |
| chestmnist     | ensemble       | ensemble        | True    |            0.0219 |             0.0219 |
| dermamnist     | ensemble       | ensemble        | True    |            0.0606 |             0.0606 |
| octmnist       | ensemble       | ensemble        | True    |            0.0668 |             0.0668 |
| organamnist    | ensemble       | ensemble        | True    |            0.005  |             0.005  |
| organcmnist    | ensemble       | ensemble        | True    |            0.0094 |             0.0094 |
| organsmnist    | ensemble       | ensemble        | True    |            0.0514 |             0.0514 |
| pathmnist      | ensemble       | ensemble        | True    |            0.0238 |             0.0238 |
| pneumoniamnist | msp            | msp             | True    |            0.0331 |             0.0331 |
| retinamnist    | ensemble       | ensemble        | True    |            0.1905 |             0.1905 |
| tissuemnist    | ensemble       | ensemble        | True    |            0.1011 |             0.1011 |

Agreement: **11/12** datasets pick the same best method under both metrics.

## coverage@risk<=0.1 by method (higher is better; 1.000 = safe at full coverage)

| dataset        |   msp |   entropy |   temp_scaling |   margin |   mc_dropout |   ensemble |   conformal_lac |   conformal_aps |
|:---------------|------:|----------:|---------------:|---------:|-------------:|-----------:|----------------:|----------------:|
| bloodmnist     | 1     |     1     |          1     |    1     |        1     |      1     |           1     |           1     |
| breastmnist    | 0.737 |     0.737 |          0.737 |    0.737 |        0.737 |      0.821 |           0     |           0.006 |
| chestmnist     | 1     |     1     |        nan     |  nan     |        1     |      1     |         nan     |         nan     |
| dermamnist     | 0.605 |     0.604 |          0.567 |    0.607 |        0.605 |      0.663 |           0.599 |           0.027 |
| octmnist       | 0.546 |     0.546 |          0.571 |    0.546 |        0.546 |      0.593 |           0.001 |           0.347 |
| organamnist    | 1     |     1     |          1     |    1     |        1     |      1     |           1     |           1     |
| organcmnist    | 1     |     1     |          1     |    1     |        1     |      1     |           1     |           1     |
| organsmnist    | 0.69  |     0.692 |          0.69  |    0.688 |        0.69  |      0.715 |           0     |           0.58  |
| pathmnist      | 0.939 |     0.936 |          0.939 |    0.938 |        0.939 |      0.942 |           0.001 |           0.898 |
| pneumoniamnist | 0.865 |     0.865 |          0.865 |    0.865 |        0.865 |      0.841 |           0.005 |           0.098 |
| retinamnist    | 0.002 |     0.05  |          0.055 |    0.002 |        0.002 |      0.15  |           0     |           0.035 |
| tissuemnist    | 0.32  |     0.32  |          0.324 |    0.32  |        0.32  |      0.406 |           0.011 |           0.194 |
