# Revision note: v1.0 -> v1.1

This note records what the first review round found, what was wrong, and what changed.
It is the audit trail for the revision and ships with the release; the manuscript
itself states the corrected conventions without reciting this history.

## Errors found in v1.0

### 1. Conformal LAC was scored by the wrong quantity (analytical error)

`abstention.conformal_lac` built the prediction set correctly (nonconformity score
`1 - p_y`, split-conformal quantile, set = `{y : 1 - p_y <= qhat}`) but then returned
`confidence = p_(1) - p_(2)` - the top-1/top-2 margin - as the abstention score. That is
a *different* object from the conformal construction. Sibling method
`conformal_aps` used `-set_size`, so the two conformal scores were compared under
inconsistent conventions, and the v1.0 claim "LAC collapses onto MSP" was an artefact
of the margin, not a property of LAC.

Evidence that this was detectable: a set-size score takes at most `K` distinct values
and has many ties, whereas the reported LAC scores were continuous and numerically
almost identical to MSP. The v1.0 code also contained this without any test asserting
the convention.

**Fix.** Both conformal scores are now scored by prediction-set size (their natural
sample-level signal). The margin is exposed as a separate, clearly labelled baseline
(`margin`). Result: with the corrected convention **both** conformal scores are far
worse than the softmax/ensemble family, and LAC is the worse of the two - the opposite
of the v1.0 claim.

### 2. The headline operating point was selected on the test set (protocol error)

`metrics.coverage_at_risk(confidence, correct, budget)` searches the risk-coverage curve
of the *same* labels it reports on. It is an oracle-selected, post-hoc statistic - like
AURC - but v1.0 described it in deployment language ("the largest coverage whose risk
stays below 10%"), which reads as a guarantee.

**Fix.** Added a second, deployable protocol (`select_threshold_on_validation` +
`evaluate_fixed_threshold`): the threshold is chosen on the validation split and applied
unchanged to the test split. On `pathmnist` this yields test coverage 0.999 at a test
risk of **0.123**, i.e. above the 10% budget: validation selection does not transfer,
so the v1.0 numbers were indeed optimistic. Both protocols are now reported side by
side and the oracle one is labelled as descriptive.

### 3. Statistical support for the ensemble claim was insufficient

v1.0 reported "best on 10/12 subsets" from point estimates, counting ties as wins.

**Fix.** `scripts/report_verdicts.py` reports, per subset: the oracle winner, the number
of methods tied at the budget, and a paired bootstrap CI for the ensemble's advantage
over the best single-model score. Corrected accounting:

| outcome | count |
| --- | --- |
| ensemble strictly best | 7 / 12 |
| ensemble tied for best | 4 / 12 |
| another method better (MSP, `pneumoniamnist`) | 1 / 12 |
| ensemble gain statistically significant (95% CI excludes 0) | 3 / 12 |

The three significant subsets are `tissuemnist` (+0.082 [+0.069, +0.096]),
`dermamnist` (+0.056 [+0.018, +0.124]) and `organsmnist` (+0.023 [+0.008, +0.038]).
The claim must be stated as "never significantly worse, significantly better on 3 of
12", not as a general superiority.

### 4. Inconsistent baseline-reproduction claim

The text claimed the backbones matched published numbers "within 0.03 on every subset"
while Table 2 showed `chestmnist` at 0.656 against a published 0.768 (gap 0.112).

**Fix.** The sentence now reports the six single-label subsets within 0.03 and states
the `chestmnist` gap explicitly, with the epoch-budget explanation.

## Regression tests added

`scripts/smoke_test.py::check_semantics` now asserts, and would fail on the v1.0 code:

1. the LAC abstention score equals `-set_size` and takes at most `K` distinct values;
2. the margin is a separate object whose ranking disagrees with the set-size ranking;
3. `coverage_at_risk` reproduces the brute-force oracle maximum (pinning its post-hoc nature);
4. the deployable protocol reports an over-budget test risk faithfully instead of clipping it.

## Consequences for the manuscript

Findings that changed: the conformal finding (2) reverses and strengthens; the ensemble
finding (1) is narrowed to "never worse, significantly better on 3 of 12"; a new
section reports the deployable protocol and the validation-to-test risk gap. The
binary-degeneracy theorem (3) stands, now with `margin` inside the equivalent group
(MSP / entropy / temperature scaling / margin), and MC dropout remains an empirical
observation, not part of the proof.

### 5. The multi-label deployable protocol was inconsistent (implementation error)

`analysis.run_analysis` selected the validation threshold for multi-label subsets with
`val_loss > 0.5` as the error indicator, while the reported risk is the mean Hamming
error of the accepted set. The two quantities disagree, so on `chestmnist` every
validation-selected threshold came out as `+inf` and the deployable row reported
coverage 0.000, which is not a measurement of anything.

**Fix.** `metrics.select_threshold_on_validation_loss` selects the threshold against
the same continuous loss that the deployable evaluation reports. `chestmnist` now
reads coverage 1.000 at test risk 0.066. The multi-label skip reasons were also
corrected: `margin` is *undefined* for multi-label subsets (top-1/top-2 has no
meaning when a sample has 14 labels), and the code used to report the generic "needs
>= 2 model bundles" reason.

### 6. Why the conformal scores fail: the calibration split is not exchangeable

Chasing the reviewer question "is APS bad, or is your implementation bad?" produced a
finding that is now Section 4.3 of the manuscript. Split-conformal coverage is a
guarantee only under exchangeability between the calibration and test data, and the
MedMNIST protocol uses the source dataset's own split where one exists, so validation
and test are *not* exchangeable for several subsets. Evidence:

- LAC in-sample calibration coverage is 0.900-0.908 on every subset, as designed, but
  test coverage falls to 0.657 (`octmnist`), 0.752 (`pneumoniamnist`), 0.782
  (`organamnist`), 0.795 (`pathmnist`); mean 0.816 against a nominal 0.900, and below
  nominal on 10 of 11 single-label subsets (`paper/NUMBERS.md`, "Validation vs test
  splits").
- Validation accuracy exceeds test accuracy by up to 0.195 (`octmnist`), and the gap
  tracks the class-prior drift between the two splits (total variation 0.316 for
  `octmnist`, 0.194 for `pathmnist`, ≤0.001 for the four subsets that do transfer).
- The same drift breaks validation-selected abstention thresholds: 54 of 80 operating
  points exceed the 10% test risk budget, with test risk up to 6.2x the validation
  estimate.

The manuscript therefore reports a post-hoc and a deployable operating point for every
score, adds the drift figure (`Figure_4_split_shift.pdf`), and states the protocol
recommendation (calibrate on data that was neither used for model selection nor
assumed to be exchangeable with test).

### 7. Control experiment: the rule itself is fine (added after the review round)

A reviewer could object that the deployable failures show a broken threshold rule rather
than non-exchangeability. `scripts/exchangeability_control.py` answers that by repeating
the identical protocol *inside* the test split: 10 random 50/50 splits per subset, each
used in both directions, so the calibration and evaluation halves are exchangeable by
construction.

| protocol | mean risk at the 0.10 budget | LAC coverage (nominal 0.900) |
| --- | --- | --- |
| validation -> test (manuscript Table 5) | 0.129 over the same five scores | 0.816 |
| exchangeable halves of the test split | **0.097** | **0.901** |

The mean calibration-to-evaluation risk gap is +0.012 (sd 0.060) and 35% of folds sit
above the budget, which is the expected marginal-guarantee noise, not a systematic
overshoot. Per subset the two protocols agree to within 0.005 exactly where the
class-prior drift is negligible and diverge where it is large (`octmnist` 0.251 vs
0.104, `organsmnist` 0.213 vs 0.100). `retinamnist` is the one subset where the control
itself overshoots (400 test samples, 200 per half, risk 0.181), i.e. small-sample noise.

This makes the failure attributive: the *split* breaks the budget, not the selection
rule. Manuscript Section 4.3, Table 6.

## Manuscript artefacts generated from the results

- `paper/tables/*.tex` (`scripts/make_latex_tables.py`) - Tables 1-6, `\input` into
  `main.tex`, so no number is hand-transcribed.
- `paper/NUMBERS.md` (`scripts/paper_numbers.py`) - every number quoted in the prose,
  including the ones that did *not* make it into a table.
- `paper/tables/table_exchangeable.tex` + `outputs/results/EXCHANGEABILITY.md`
  (`scripts/exchangeability_control.py`) - the Step A control of Revision item 7.
- `paper/baselines_medmnist_v2.csv` - the published MedMNIST v2 ResNet-18 (28) values
  used in Table 2, transcribed from Table 3 of arXiv:2110.14795v2.

## Still to do

- Optional: evaluate 224x224 (blocked by the 20 GB Kaggle disk), add a second
  backbone, or add a calibration split carved out of the training data.
