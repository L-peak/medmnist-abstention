# When Should a Medical Image Classifier Abstain? — benchmark release (v1.1)

Code, results and logit bundles for a cross-modality benchmark of **abstention
(selective prediction)** scores on medical image classifiers: twelve MedMNIST v2 2D
subsets, eleven diagnostic modalities, eight abstention scores, three seeds at
$28\times28$ and two seeds at $128\times128$ on five subsets.

Manuscript: *When Should a Medical Image Classifier Abstain? A Cross-Modality Benchmark
of Selective Prediction* (submitted to Computer Methods and Programs in Biomedicine,
2026).

## What is in this repository

| Path | Contents |
| --- | --- |
| `src/`, `scripts/` | Training and analysis pipeline. Every analysis step runs on CPU from the released logits, so no GPU is needed to reproduce the results. |
| `outputs/logits/` | 48 logit bundles (validation/test logits, labels, MC-dropout moments) exported by the training runs. |
| `outputs/results/` | Per-subset metrics, risk--coverage curves, the exchangeability control, generated LaTeX tables. |
| `outputs/figures/` | Per-subset risk--coverage figures. |
| `paper/` | `NUMBERS.md` (every number quoted in the manuscript), `REVISION_NOTE.md` (what changed between v1.0 and v1.1 and why), generated tables, the published MedMNIST baselines used for the sanity check. |
| `kaggle/` | Batch runner used on the free Kaggle T4$\times$2 tier (resumable, time-budgeted). |

## Headline results (v1.1)

- **Deep ensembles** are best or tied for best on 11 of 12 subsets and never
  significantly worse, but significantly better than the best single-model score on
  only 3 of 12 subsets (and on none at $128\times128$).
- **Both conformal scores** (least ambiguous classifier and adaptive prediction sets,
  each scored by prediction-set size) are the weakest of the eight on every subset where
  the scores separate at all.
- **Validation-calibrated thresholds do not transfer**: 54 of 80 operating points exceed
  the 10% test risk budget, with test risk up to 6.2$\times$ the validation estimate,
  because the official MedMNIST validation and test splits are not exchangeable.
- **Exchangeability control**: with exchangeable halves of the test split, the same
  threshold rule meets its budget (mean risk 0.097) and split-conformal LAC recovers
  nominal coverage (0.901 against 0.900).
- **Binary degeneracy**: on binary tasks MSP, entropy, temperature scaling and the
  top-1/top-2 margin are the same ranker by proof, so comparisons on binary data alone
  cannot support conclusions about them.

## Reproduce (CPU only)

```bash
python scripts/smoke_test.py                  # pins the scoring conventions
python scripts/run_selective.py --stem pathmnist_resnet18_28
python scripts/summarize_results.py --size 28
python scripts/report_verdicts.py --size 28   # ties, verdicts, bootstrap CIs
python scripts/exchangeability_control.py     # control experiment (~90 s)
python scripts/make_paper_figures.py --figure all
python scripts/make_latex_tables.py           # paper/tables/*.tex
python scripts/paper_numbers.py               # paper/NUMBERS.md
python scripts/verify_paper_claims.py         # 15 independent assertions
python scripts/submission_audit.py            # 19 structural checks
python scripts/verify_references.py
```

## Data

All datasets are MedMNIST v2 (CC BY 4.0); please cite
[Yang et al., *Scientific Data* 10, 2023](https://doi.org/10.1038/s41597-022-01721-8).
The logit bundles in `outputs/logits/` are derived from MedMNIST and are redistributed
under the same terms.

## License

Code: MIT (see `LICENSE`). Logit bundles: derived from MedMNIST v2, CC BY 4.0.

## Citation

See `CITATION.cff`.
