# Numbers quoted in the manuscript (generated)

Source: outputs/results/*.csv and *.json under D:\HWB\S1\medmnist-abstention\outputs\results


## Table 2 - backbone sanity check (28x28, mean of 3 seeds)

| subset | ours | published ResNet-18 (28) | source metric | gap | within 0.03 |
| --- | --- | --- | --- | --- | --- |
| pathmnist | 0.898 | 0.907 | accuracy | -0.009 | yes |
| chestmnist | 0.656 | 0.768 | macro-AUC | -0.112 | NO |
| dermamnist | 0.751 | 0.735 | accuracy | +0.016 | yes |
| octmnist | 0.754 | 0.743 | accuracy | +0.011 | yes |
| pneumoniamnist | 0.853 | 0.854 | accuracy | -0.001 | yes |
| retinamnist | 0.472 | 0.524 | accuracy | -0.052 | NO |
| breastmnist | 0.842 | 0.863 | accuracy | -0.021 | yes |
| bloodmnist | 0.958 | 0.958 | accuracy | +0.000 | yes |
| tissuemnist | 0.668 | 0.676 | accuracy | -0.008 | yes |
| organamnist | 0.934 | 0.935 | accuracy | -0.001 | yes |
| organcmnist | 0.913 | 0.900 | accuracy | +0.013 | yes |
| organsmnist | 0.776 | 0.782 | accuracy | -0.006 | yes |

within 0.03: 10/12; outside: chestmnist -0.112, retinamnist -0.052
- chestmnist: ours 0.656 (macro-AUC) vs published 0.768, gap -0.112
- retinamnist: ours 0.472 (accuracy) vs published 0.524, gap -0.052

own test accuracy per subset (28x28, mean of seeds, from SUMMARY.md):
- bloodmnist: 0.958 (3 seeds)
- breastmnist: 0.842 (3 seeds)
- chestmnist: 0.448 (3 seeds, macro-AUC 0.656)
- dermamnist: 0.751 (3 seeds)
- octmnist: 0.754 (3 seeds)
- organamnist: 0.934 (3 seeds)
- organcmnist: 0.913 (3 seeds)
- organsmnist: 0.776 (3 seeds)
- pathmnist: 0.898 (3 seeds)
- pneumoniamnist: 0.853 (3 seeds)
- retinamnist: 0.472 (3 seeds)
- tissuemnist: 0.668 (3 seeds)

## Coverage at risk<=0.1 (28x28)

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

per-subset oracle winner, ties, ensemble advantage:
- bloodmnist: best=mc_dropout (1.000), verdict=ensemble_tied_for_best, tied=8, ens-vs-best-single +0.000 [+0.000, +0.000]
- breastmnist: best=ensemble (0.821), verdict=ensemble_strictly_best, tied=1, ens-vs-best-single +0.083 [-0.090, +0.295]
- chestmnist: best=ensemble (1.000), verdict=ensemble_tied_for_best, tied=4, ens-vs-best-single n/a
- dermamnist: best=ensemble (0.663), verdict=ensemble_strictly_best, tied=1, ens-vs-best-single +0.056 [+0.018, +0.124]
- octmnist: best=ensemble (0.593), verdict=ensemble_strictly_best, tied=1, ens-vs-best-single +0.047 [-0.024, +0.127]
- organamnist: best=ensemble (1.000), verdict=ensemble_tied_for_best, tied=8, ens-vs-best-single +0.000 [+0.000, +0.000]
- organcmnist: best=ensemble (1.000), verdict=ensemble_tied_for_best, tied=8, ens-vs-best-single +0.000 [+0.000, +0.000]
- organsmnist: best=ensemble (0.715), verdict=ensemble_strictly_best, tied=1, ens-vs-best-single +0.023 [+0.008, +0.038]
- pathmnist: best=ensemble (0.942), verdict=ensemble_strictly_best, tied=1, ens-vs-best-single +0.002 [-0.006, +0.013]
- pneumoniamnist: best=msp (0.865), verdict=another_method_better, tied=5, ens-vs-best-single -0.024 [-0.040, +0.032]
- retinamnist: best=ensemble (0.150), verdict=ensemble_strictly_best, tied=1, ens-vs-best-single +0.095 [-0.033, +0.242]
- tissuemnist: best=ensemble (0.406), verdict=ensemble_strictly_best, tied=1, ens-vs-best-single +0.082 [+0.069, +0.096]

strict 7/12, tied 4, lost 1
ensemble advantage CI excludes 0 on: dermamnist, organsmnist, tissuemnist (3/12)

## Validation vs test splits (seed 0 bundles)

| subset | n_val | n_test | val acc | test acc | gap | val risk | test risk | test/val risk | class-prior L1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bloodmnist | 1712 | 3421 | 0.960 | 0.956 | +0.004 | 0.040 | 0.044 | 1.1 | 0.001 |
| breastmnist | 78 | 156 | 0.821 | 0.827 | -0.006 | 0.179 | 0.173 | 1.0 | 0.000 |
| chestmnist | 11219 | 22433 | 0.935 | 0.934 | +0.001 | 0.065 | 0.066 | 1.0 | nan |
| dermamnist | 1003 | 2005 | 0.762 | 0.750 | +0.012 | 0.238 | 0.250 | 1.1 | 0.001 |
| octmnist | 10832 | 1000 | 0.943 | 0.748 | +0.195 | 0.057 | 0.252 | 4.4 | 0.316 |
| organamnist | 6491 | 17778 | 0.984 | 0.932 | +0.052 | 0.016 | 0.068 | 4.2 | 0.131 |
| organcmnist | 2392 | 8216 | 0.974 | 0.916 | +0.058 | 0.026 | 0.084 | 3.2 | 0.191 |
| organsmnist | 2452 | 8827 | 0.894 | 0.779 | +0.115 | 0.106 | 0.221 | 2.1 | 0.166 |
| pathmnist | 10004 | 7180 | 0.951 | 0.876 | +0.075 | 0.049 | 0.124 | 2.6 | 0.194 |
| pneumoniamnist | 524 | 624 | 0.977 | 0.859 | +0.118 | 0.023 | 0.141 | 6.2 | 0.117 |
| retinamnist | 120 | 400 | 0.442 | 0.487 | -0.046 | 0.558 | 0.512 | 0.9 | 0.018 |
| tissuemnist | 23640 | 47280 | 0.666 | 0.666 | -0.000 | 0.334 | 0.334 | 1.0 | 0.000 |

largest risk inflation: pneumoniamnist val risk 0.023 -> test risk 0.141 (6.2x)

## Control experiment: exchangeable halves of the test split

- folds: 10 random splits x 2 directions per subset
- deployable protocol: mean evaluation-half risk 0.097 (budget 0.1), mean risk gap +0.012, folds over budget 0.35
- conformal_lac: mean evaluation-half coverage 0.901 (nominal 0.90)
- conformal_aps: mean evaluation-half coverage 0.972 (nominal 0.90)

per-subset mean evaluation-half risk:
| dataset        |   risk_out |
|:---------------|-----------:|
| bloodmnist     |      0.044 |
| breastmnist    |      0.101 |
| chestmnist     |      0.058 |
| dermamnist     |      0.1   |
| octmnist       |      0.104 |
| organamnist    |      0.068 |
| organcmnist    |      0.084 |
| organsmnist    |      0.1   |
| pathmnist      |      0.1   |
| pneumoniamnist |      0.103 |
| retinamnist    |      0.187 |
| tissuemnist    |      0.1   |

validation -> test risk for comparison (same five scores): see the table under 'Deployable protocol' above

## AURC vs AUGRC winner agreement (28x28)

- bloodmnist: AURC->mc_dropout, AUGRC->ensemble  <-- differs
- breastmnist: AURC->ensemble, AUGRC->ensemble
- chestmnist: AURC->ensemble, AUGRC->ensemble
- dermamnist: AURC->ensemble, AUGRC->ensemble
- octmnist: AURC->ensemble, AUGRC->ensemble
- organamnist: AURC->ensemble, AUGRC->ensemble
- organcmnist: AURC->ensemble, AUGRC->ensemble
- organsmnist: AURC->ensemble, AUGRC->ensemble
- pathmnist: AURC->ensemble, AUGRC->ensemble
- pneumoniamnist: AURC->msp, AUGRC->msp
- retinamnist: AURC->ensemble, AUGRC->ensemble
- tissuemnist: AURC->ensemble, AUGRC->ensemble

agreement 11/12

## AURC vs AUGRC winner agreement (128x128)

- bloodmnist: AURC->mc_dropout, AUGRC->mc_dropout
- dermamnist: AURC->ensemble, AUGRC->ensemble
- octmnist: AURC->ensemble, AUGRC->ensemble
- pathmnist: AURC->ensemble, AUGRC->ensemble
- retinamnist: AURC->ensemble, AUGRC->ensemble

agreement 5/5

## Deployable protocol (threshold from validation, evaluated on test)

| dataset        |   scores | excluded   |   median_test_risk |   min_test_risk |   max_test_risk |   over_budget |   median_test_coverage |
|:---------------|---------:|:-----------|-------------------:|----------------:|----------------:|--------------:|-----------------------:|
| bloodmnist     |        7 | mc_dropout |              0.044 |           0.044 |           0.044 |             0 |                  1     |
| breastmnist    |        7 | mc_dropout |              0.157 |           0.087 |           0.333 |             6 |                  0.897 |
| chestmnist     |        3 | mc_dropout |              0.066 |           0.066 |           0.066 |             0 |                  1     |
| dermamnist     |        7 | mc_dropout |              0.111 |           0.109 |           0.177 |             7 |                  0.633 |
| octmnist       |        7 | mc_dropout |              0.252 |           0.251 |           0.252 |             7 |                  1     |
| organamnist    |        7 | mc_dropout |              0.067 |           0.067 |           0.068 |             0 |                  0.999 |
| organcmnist    |        7 | mc_dropout |              0.084 |           0.083 |           0.084 |             0 |                  1     |
| organsmnist    |        7 | mc_dropout |              0.214 |           0.209 |           0.221 |             7 |                  0.985 |
| pathmnist      |        7 | mc_dropout |              0.124 |           0.123 |           0.124 |             7 |                  1     |
| pneumoniamnist |        7 | mc_dropout |              0.138 |           0.138 |           0.141 |             7 |                  0.99  |
| retinamnist    |        7 | mc_dropout |              0.208 |           0.184 |           0.271 |             7 |                  0.252 |
| tissuemnist    |        7 | mc_dropout |              0.101 |           0.098 |           0.12  |             6 |                  0.323 |

scores landing above the 10% budget on test: 54/80 (68%); subsets where every score is over budget: dermamnist, octmnist, organsmnist, pathmnist, pneumoniamnist, retinamnist
subsets where no score is over budget: bloodmnist, chestmnist, organamnist, organcmnist

## Conformal diagnostics (marginal coverage and set size)


conformal_lac: mean set size 1.30, empirical coverage 0.816 (target >= 0.90); below target on 10/11: bloodmnist, breastmnist, dermamnist, octmnist, organamnist, organcmnist, organsmnist, pathmnist, pneumoniamnist, retinamnist
| dataset        |   mean_set_size |   empirical_coverage |
|:---------------|----------------:|---------------------:|
| bloodmnist     |           0.912 |                0.897 |
| breastmnist    |           1.096 |                0.859 |
| dermamnist     |           1.719 |                0.897 |
| octmnist       |           0.797 |                0.657 |
| organamnist    |           0.788 |                0.782 |
| organcmnist    |           0.802 |                0.788 |
| organsmnist    |           1.009 |                0.784 |
| pathmnist      |           0.856 |                0.795 |
| pneumoniamnist |           0.814 |                0.752 |
| retinamnist    |           3.078 |                0.865 |
| tissuemnist    |           2.375 |                0.904 |

conformal_aps: mean set size 3.92, empirical coverage 0.962 (target >= 0.90); below target on 2/11: pneumoniamnist, retinamnist
| dataset        |   mean_set_size |   empirical_coverage |
|:---------------|----------------:|---------------------:|
| bloodmnist     |           4.144 |                0.999 |
| breastmnist    |           1.962 |                0.987 |
| dermamnist     |           3.292 |                0.954 |
| octmnist       |           2.552 |                0.96  |
| organamnist    |           6.165 |                1     |
| organcmnist    |           5.863 |                0.998 |
| organsmnist    |           6.314 |                0.992 |
| pathmnist      |           4.166 |                0.985 |
| pneumoniamnist |           1     |                0.859 |
| retinamnist    |           3.63  |                0.888 |
| tissuemnist    |           4.03  |                0.964 |

## Binary subsets: how many scores coincide exactly

- pneumoniamnist: best coverage 0.865; tied at the optimum: entropy, margin, mc_dropout, msp, temp_scaling
- breastmnist: best coverage 0.821; tied at the optimum: ensemble

pneumoniamnist - coverage_at_risk and AURC per score:
| method        |   coverage_at_risk |   aurc |
|:--------------|-------------------:|-------:|
| msp           |             0.8654 | 0.047  |
| entropy       |             0.8654 | 0.047  |
| temp_scaling  |             0.8654 | 0.047  |
| margin        |             0.8654 | 0.047  |
| mc_dropout    |             0.8654 | 0.0472 |
| ensemble      |             0.8413 | 0.0477 |
| conformal_aps |             0.0978 | 0.1379 |
| conformal_lac |             0.0048 | 0.2402 |

breastmnist - coverage_at_risk and AURC per score:
| method        |   coverage_at_risk |   aurc |
|:--------------|-------------------:|-------:|
| ensemble      |             0.8205 | 0.0556 |
| msp           |             0.7372 | 0.072  |
| temp_scaling  |             0.7372 | 0.072  |
| entropy       |             0.7372 | 0.072  |
| margin        |             0.7372 | 0.072  |
| mc_dropout    |             0.7372 | 0.072  |
| conformal_lac |             0      | 0.162  |
| conformal_aps |             0.0064 | 0.182  |

## Futility regime

- retinamnist: test accuracy 0.472; best AURC 0.2941 (ensemble); best coverage@0.1 0.150; median coverage across scores 0.019
- chestmnist: test accuracy 0.448; best AURC 0.0353 (ensemble); best coverage@0.1 1.000; median coverage across scores 1.000
  exact-match risk summary present: null

chestmnist keys: ['best_coverage', 'best_method', 'best_method_verdict', 'class_conditional_example', 'ensemble_advantage', 'exact_match_correct_rate', 'hamming_risk_at_full_coverage', 'macro_auc', 'meta', 'methods_run', 'methods_skipped', 'metric_note', 'n_test', 'operating_point', 'sources', 'subset_accuracy', 'test_accuracy', 'tied_methods']

chestmnist under an exact-match risk definition: exact-match accuracy 0.447; coverage@0.1 = 0.000 (vs 1.000 under Hamming risk)

## Coverage at risk<=0.1 (128x128)

| dataset     |   msp |   entropy |   temp_scaling |   margin |   mc_dropout |   ensemble |   conformal_lac |   conformal_aps |
|:------------|------:|----------:|---------------:|---------:|-------------:|-----------:|----------------:|----------------:|
| bloodmnist  | 1     |     1     |          1     |    1     |        1     |      1     |           1     |           1     |
| dermamnist  | 0.652 |     0.651 |          0.635 |    0.656 |        0.652 |      0.663 |           0.624 |           0.285 |
| octmnist    | 0.901 |     0.904 |          0.902 |    0.901 |        0.901 |      0.902 |           0     |           0.445 |
| pathmnist   | 1     |     1     |          1     |    1     |        1     |      1     |           1     |           1     |
| retinamnist | 0.13  |     0.125 |          0.152 |    0.13  |        0.13  |      0.132 |           0.112 |           0.025 |

per-subset oracle winner, ties, ensemble advantage:
- bloodmnist: best=mc_dropout (1.000), verdict=ensemble_tied_for_best, tied=8, ens-vs-best-single +0.000 [+0.000, +0.000]
- dermamnist: best=ensemble (0.663), verdict=ensemble_strictly_best, tied=1, ens-vs-best-single +0.012 [-0.014, +0.044]
- octmnist: best=ensemble (0.904), verdict=another_method_better, tied=1, ens-vs-best-single +0.001 [-0.037, +0.025]
- pathmnist: best=ensemble (1.000), verdict=ensemble_tied_for_best, tied=8, ens-vs-best-single +0.000 [+0.000, +0.000]
- retinamnist: best=ensemble (0.152), verdict=another_method_better, tied=1, ens-vs-best-single -0.020 [-0.085, +0.150]

strict 1/5, tied 2, lost 2
ensemble advantage CI excludes 0 on:  (0/5)
