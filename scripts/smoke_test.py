"""End-to-end sanity check on synthetic data.

Verifies: data -> training -> evaluation -> every abstention method -> risk-coverage metrics.
Requires only torch/numpy/pandas (no medmnist, no network, no GPU).
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import abstention as abst  # noqa: E402
from src.config import ABSTENTION_METHODS, SEED  # noqa: E402
from src.data import synthetic_loaders  # noqa: E402
from src.metrics import (  # noqa: E402
    augrc,
    augrc_closed_form,
    aurc,
    class_conditional_coverage,
    clinical_translation,
    coverage_at_risk,
    e_aurc,
    ece,
    evaluate_fixed_threshold,
    oracle_aurc,
    risk_coverage_curve,
    select_threshold_on_validation,
)
from src.models import build_model, resolve_device  # noqa: E402
from src.train import evaluate, seed_everything, train_model  # noqa: E402


def check_metrics() -> None:
    """Hand-checkable cases so a silent regression cannot pass."""
    correct = np.array([1, 1, 0, 0], dtype=bool)
    confidence = np.array([0.9, 0.8, 0.2, 0.1])
    coverage, risk = risk_coverage_curve(confidence, correct)
    assert np.allclose(coverage, [0.25, 0.5, 0.75, 1.0]), coverage
    assert np.allclose(risk, [0.0, 0.0, 1 / 3, 0.5]), risk
    assert 0.0 <= aurc(confidence, correct) <= oracle_aurc(correct) + 1e-9
    assert e_aurc(confidence, correct) >= -1e-9
    # Perfect ranking reaches the oracle floor.
    assert abs(aurc(np.array([2.0, 1.0, 0.5, 0.0]), correct) - oracle_aurc(correct)) < 1e-9
    assert coverage_at_risk(confidence, correct, 0.0) == 0.5
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])
    assert ece(probs, np.array([0, 0])) > 0
    print("[ok] metrics unit checks")


def check_augrc(confidence, correct) -> None:
    """The curve integral must reproduce the published closed form (Eq. 7)."""
    curve = augrc(confidence, correct=correct)
    closed = augrc_closed_form(confidence, correct)
    assert abs(curve - closed) < 5e-3, (curve, closed)
    # Deterministic ordering check: ranking every failure last must beat a random
    # ranker. (Asserting this for a model's own confidence is unreliable here: the
    # synthetic test set has ~10 errors, so its AUROC estimate is dominated by noise.)
    perfect = np.asarray(correct).astype(float)
    rng = np.random.default_rng(0)
    random_scores = rng.random(perfect.size)
    a_perfect = augrc(perfect, correct=correct)
    a_random = augrc(random_scores, correct=correct)
    assert a_perfect < a_random, (a_perfect, a_random)
    print(f"[ok] AUGRC curve integral {curve:.4f} == closed form {closed:.4f}; "
          f"perfect ranker {a_perfect:.4f} < random {a_random:.4f} "
          f"(model's own score {augrc(confidence, correct=correct):.4f}, noisy at n={perfect.size})")


def check_semantics() -> None:
    """Semantic checks for the two mistakes found in review (v1.0 -> v1.1).

    v1.0 scored conformal LAC by the top-1/top-2 margin instead of the conformal
    prediction-set size, and reported a test-set-selected operating point as if it were
    deployable. These assertions fail on the v1.0 code.
    """
    rng = np.random.default_rng(3)
    n_classes, n_cal, n_test, budget = 5, 400, 200, 0.1
    probs_cal = abst.predict_proba(rng.normal(0, 1.2, (n_cal, n_classes)))
    labels_cal = rng.integers(0, n_classes, n_cal)
    probs_test = abst.predict_proba(rng.normal(0, 1.2, (n_test, n_classes)))

    lac = abst.conformal_lac(probs_cal, labels_cal, probs_test, alpha=budget)

    # (1) LAC's abstention score must be the conformal set size, i.e. a step function
    #     with at most K distinct values - never a continuous margin.
    assert np.allclose(lac["confidence"], -np.asarray(lac["set_size"], dtype=float)), \
        "LAC abstention score is not the prediction-set size"
    n_unique = np.unique(lac["confidence"]).size
    assert n_unique <= n_classes, f"set-size score has {n_unique} values for K={n_classes}"
    assert np.unique(lac["margin"]).size > n_unique, "margin should be continuous-valued"
    print(f"[ok] LAC abstention score == -set_size ({n_unique} distinct values, K={n_classes}); "
          f"margin kept separate ({np.unique(lac['margin']).size} distinct values)")

    # (2) the margin ranking is a different object and must be reported as its own baseline
    rank_margin = np.argsort(-np.asarray(lac["margin"]))
    rank_setsize = np.argsort(-np.asarray(lac["confidence"]), kind="mergesort")
    agreement = float(np.mean(rank_margin == rank_setsize))
    assert agreement < 1.0, "margin and set-size rankings coincide; test is not discriminating"
    print(f"[ok] margin and set-size rankings differ (element-wise agreement {agreement:.2f})")

    # (3) coverage_at_risk is an oracle statistic: it searches thresholds on the same
    #     labels it is scored on. Pin that property down so it cannot be mistaken for a
    #     deployable operating point.
    correct = rng.random(n_test) < 0.8
    confidence = rng.random(n_test)
    oracle = coverage_at_risk(confidence, correct, budget)
    brute = 0.0
    for tau in np.unique(confidence):
        accept = confidence >= tau
        if accept.any() and (1.0 - correct[accept].mean()) <= budget:
            brute = max(brute, float(accept.mean()))
    assert abs(oracle - brute) < 1e-12, (oracle, brute)

    conf_val, correct_val = rng.random(300), rng.random(300) < 0.8
    tau = select_threshold_on_validation(conf_val, correct_val, budget)
    _, risk_val = evaluate_fixed_threshold(conf_val, correct_val, tau)
    assert risk_val <= budget + 1e-12, "validation-selected threshold must meet the budget on validation"
    print(f"[ok] coverage@risk is oracle-selected (== brute-force max {brute:.3f}); "
          f"validation-selected threshold has validation risk {risk_val:.3f}")

    # (4) the deployable protocol must report an over-budget test risk honestly
    conf_test, correct_test = rng.random(n_test), rng.random(n_test) < 0.5
    coverage_fixed, risk_fixed = evaluate_fixed_threshold(conf_test, correct_test, tau)
    assert risk_fixed > budget, "constructed case should overshoot the budget on test"
    print(f"[ok] deployable protocol reports test risk {risk_fixed:.3f} at coverage "
          f"{coverage_fixed:.3f} instead of clipping it to the {budget:.1f} budget")


def main() -> int:
    seed_everything(SEED)
    device = resolve_device(None)
    train_loader, val_loader, test_loader, meta = synthetic_loaders(n_classes=3, size=28, batch_size=64)
    print(f"device={device} synthetic task={meta.task} classes={meta.n_classes}")

    model = build_model("smallcnn", meta.n_channels, meta.n_classes, meta.size).to(device)
    train_model(model, train_loader, val_loader, meta, device, epochs=6, lr=3e-3, verbose=True)

    val_logits, val_labels = evaluate(model, val_loader, meta, device)
    test_logits, test_labels = evaluate(model, test_loader, meta, device)
    correct = test_logits.argmax(axis=1) == test_labels
    val_correct = val_logits.argmax(axis=1) == val_labels
    print(f"val accuracy: {val_correct.mean():.4f} | test accuracy: {correct.mean():.4f}")
    assert correct.size == len(test_labels)
    # The synthetic task is separable by construction; if val/test are near chance the
    # task/splits are broken (this is exactly the bug the shared-centre design prevents).
    assert val_correct.mean() > 0.6, f"val accuracy {val_correct.mean():.3f} - synthetic task is not learnable"
    assert correct.mean() > 0.6, f"test accuracy {correct.mean():.3f} - synthetic task is not learnable"

    results = {}
    for name in ABSTENTION_METHODS:
        if name in ("msp", "entropy"):
            out = getattr(abst, name)(test_logits)
        elif name == "margin":
            out = abst.margin_confidence(test_logits)
        elif name == "temp_scaling":
            out = abst.temp_scaling(val_logits, val_labels, test_logits)
            assert 0.1 < out["temperature"] < 50, out["temperature"]
        elif name == "mc_dropout":
            out = abst.mc_dropout(model, test_loader, device, n_samples=5)
        elif name == "ensemble":
            out = abst.ensemble_confidence([abst.predict_proba(test_logits) for _ in range(3)])
        elif name == "conformal_lac":
            out = abst.conformal_lac(abst.predict_proba(val_logits), val_labels, abst.predict_proba(test_logits), alpha=0.1)
            assert 0.75 <= abst.empirical_coverage(out["include"], test_labels) <= 1.0, "LAC coverage out of range"
        elif name == "conformal_aps":
            out = abst.conformal_aps(abst.predict_proba(val_logits), val_labels, abst.predict_proba(test_logits), alpha=0.1)
            assert 0.75 <= abst.empirical_coverage(out["include"], test_labels) <= 1.0, "APS coverage out of range"
        else:
            raise ValueError(name)

        conf = np.asarray(out["confidence"], dtype=float)
        assert conf.shape == (test_labels.size,), f"{name}: bad confidence shape {conf.shape}"
        assert np.isfinite(conf).all(), f"{name}: non-finite confidence"
        score = aurc(conf, correct)
        assert np.isfinite(score)
        results[name] = {
            "aurc": score,
            "e_aurc": e_aurc(conf, correct),
            "coverage_at_risk": coverage_at_risk(conf, correct, 0.1),
        }

    print("\nmethod            AURC     E-AURC   cov@risk")
    for name, row in sorted(results.items(), key=lambda kv: kv[1]["aurc"]):
        print(f"{name:16s} {row['aurc']:.4f}  {row['e_aurc']:.4f}  {row['coverage_at_risk']:.3f}")

    msp_conf = abst.msp(test_logits)["confidence"]
    print("\nper-class acceptance (median-confidence threshold):")
    for cls, stats in class_conditional_coverage(msp_conf, correct, test_labels, float(np.median(msp_conf))).items():
        print(f"  class {cls}: n={stats['n']:4d} coverage={stats['coverage']:.3f} risk={stats['risk_on_accepted']:.3f}")
    print("\nclinical translation @ full coverage:", clinical_translation(1.0 - float(correct.mean()), 1.0))

    # --- stage 2: the GPU -> CPU hand-off must survive a round trip ---------------
    from src.analysis import LogitBundle, load_bundle, run_analysis, save_bundle
    from src.config import OUT_DIR

    mc = abst.mc_dropout(model, test_loader, device, n_samples=5)
    mc_probs, mc_var = mc["probs"], mc["probs_var"]
    bundle_a = LogitBundle(val_logits, val_labels, test_logits, test_labels,
                           mc_probs=mc_probs, mc_var=mc_var, meta=vars(meta), source="synthetic_seed0")
    bundle_b = LogitBundle(val_logits + 0.1, val_labels, test_logits + 0.1, test_labels,
                           mc_probs=mc_probs, mc_var=mc_var, meta=vars(meta), source="synthetic_seed1")
    path_a = save_bundle(OUT_DIR / "smoke_bundle_seed0.npz", bundle_a)
    path_b = save_bundle(OUT_DIR / "smoke_bundle_seed1.npz", bundle_b)
    loaded = [load_bundle(path_a), load_bundle(path_b)]
    assert np.allclose(loaded[0].test_logits, test_logits, atol=1e-4), "bundle round trip lost values"

    table, curves, summary, skipped = run_analysis(loaded, alpha=0.1, max_risk=0.25)
    assert skipped == [], f"unexpected skipped methods: {skipped}"
    assert len(table) == len(ABSTENTION_METHODS), (len(table), len(ABSTENTION_METHODS))
    assert np.isfinite(table["aurc"]).all()
    direct = aurc(abst.msp(test_logits)["confidence"], correct)
    from_bundle = float(table.loc[table["method"] == "msp", "aurc"].iloc[0])
    assert abs(direct - from_bundle) < 1e-6, (direct, from_bundle)
    print("\n[ok] logit-bundle round trip: analysis reproduces the direct computation")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"[ok] best={summary['best_method']} operating point: coverage={summary['operating_point']['coverage']:.3f}")

    # Plotting must run in its own process: loading torch and matplotlib into one
    # process aborts on Windows/conda with a duplicate libiomp5md.dll error.
    curves_path = OUT_DIR / "smoke_risk_coverage_input.npz"
    np.savez_compressed(curves_path, **curves)
    figure = OUT_DIR / "smoke_risk_coverage.png"
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("plot_risk_coverage.py")),
         "--npz", str(curves_path), "--out", str(figure), "--title", "synthetic smoke test"],
        check=True,
    )
    assert figure.exists() and figure.stat().st_size > 0, "figure not written"
    print(f"[ok] risk-coverage figure via separate process: {figure}")

    # A single bundle must honestly skip the ensemble instead of faking one.
    _, _, _, skipped_single = run_analysis([loaded[0]], max_risk=0.25)
    assert [s["method"] for s in skipped_single] == ["ensemble"], skipped_single
    print("[ok] single-bundle run reports ensemble as skipped (no silent fallback)")

    # --- stage 3: multi-label path (chestmnist style, 14 labels) ------------------
    from src.abstention import multilabel_confidence, multilabel_ensemble
    from src.metrics import macro_auc, subset_accuracy

    rng = np.random.default_rng(0)
    n_labels, n_samples = 14, 400
    labels_ml = (rng.random((n_samples, n_labels)) < 0.3).astype(int)
    logits_ml = rng.normal(0.0, 1.0, size=(n_samples, n_labels)) + labels_ml * 1.0
    probs_ml = 1.0 / (1.0 + np.exp(-logits_ml))

    auc = macro_auc(labels_ml, probs_ml)
    subset = subset_accuracy(labels_ml, probs_ml)
    assert 0.5 < auc <= 1.0, auc
    assert 0.0 <= subset <= 1.0, subset
    print(f"[ok] multi-label metrics: macro_auc={auc:.3f} subset_acc={subset:.3f}")

    ml_meta = {"task": "multi-label", "dataset": "synthetic-ml", "n_classes": n_labels}
    ml_bundles = [
        LogitBundle(logits_ml, labels_ml, logits_ml + shift, labels_ml,
                    mc_probs=probs_ml, meta=ml_meta, source=f"synthetic-ml-seed{k}")
        for k, shift in enumerate((0.0, 0.1))
    ]
    ml_table, ml_curves, ml_summary, ml_skipped = run_analysis(ml_bundles, max_risk=0.2)
    ran = sorted(ml_table["method"])
    assert ran == ["ensemble", "entropy", "mc_dropout", "msp"], ran
    skipped_names = sorted(item["method"] for item in ml_skipped)
    # `margin` is undefined for multi-label targets (there is no top-2 class margin).
    assert skipped_names == ["conformal_aps", "conformal_lac", "margin", "temp_scaling"], skipped_names
    assert "multi-label" in ml_summary["metric_note"]
    assert np.isfinite(ml_table["aurc"]).all()
    assert "macro_auc" in ml_summary
    print(f"[ok] multi-label analysis: ran={ran} skipped={skipped_names}")
    print(ml_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    check_metrics()
    check_augrc(msp_conf, correct)
    check_semantics()
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
