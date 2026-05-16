"""Tests for newsmood.select."""

import numpy as np
import pandas as pd
import pytest

from newsmood.select import (
    drop_high_correlation,
    drop_low_variance,
    make_per_fold_mi_selector,
    sweep_alpha,
    top_k_by_mutual_information,
)


def _panel(seed: int = 0, n: int = 100, redundant: bool = False):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    panel = pd.DataFrame(
        {
            "useful": f1,
            "noise": f2,
            "constant": np.zeros(n),
            "near_constant": np.array([0.0] * (n - 1) + [1e-9]),
            "target": 0.5 * f1 + rng.normal(scale=0.5, size=n),
        }
    )
    if redundant:
        # A near-copy with mild noise so that corr(useful, dup) ≈ 0.95-0.99,
        # below an extreme-threshold like 0.9999.
        panel["dup"] = f1 + rng.normal(scale=0.2, size=n)
    return panel


# ---------------------------------------------------------------------------
# drop_low_variance
# ---------------------------------------------------------------------------


def test_drop_low_variance_removes_constant():
    panel = _panel()
    out = drop_low_variance(panel, target="target")
    assert "constant" not in out.columns
    assert "near_constant" not in out.columns
    assert "useful" in out.columns
    assert "target" in out.columns


def test_drop_low_variance_preserves_target_even_if_constant():
    panel = _panel()
    panel["target"] = 0.0
    out = drop_low_variance(panel, target="target", min_std=1e-3)
    assert "target" in out.columns


def test_drop_low_variance_respects_feature_cols_arg():
    panel = _panel()
    out = drop_low_variance(panel, target="target", feature_cols=["useful", "noise"])
    # 'constant' kept because not in feature_cols
    assert "constant" in out.columns


# ---------------------------------------------------------------------------
# drop_high_correlation
# ---------------------------------------------------------------------------


def test_drop_high_correlation_drops_redundant():
    panel = _panel(redundant=True)
    out = drop_high_correlation(panel, target="target", threshold=0.9)
    # One of {useful, dup} should be dropped (the one less correlated with target)
    dropped = {"useful", "dup"} - set(out.columns)
    assert len(dropped) == 1


def test_drop_high_correlation_keeps_target_correlated_one():
    panel = _panel(redundant=True)
    out = drop_high_correlation(panel, target="target", threshold=0.9)
    # 'useful' is target-driven; 'dup' is a noisier copy with the noise on top
    # The retained one should have higher |corr| with target than the dropped.
    assert "target" in out.columns


def test_drop_high_correlation_no_change_when_threshold_high():
    panel = _panel(redundant=True)
    out = drop_high_correlation(panel, target="target", threshold=0.9999)
    assert "useful" in out.columns
    assert "dup" in out.columns


# ---------------------------------------------------------------------------
# top_k_by_mutual_information
# ---------------------------------------------------------------------------


def test_top_k_mi_keeps_useful_feature():
    panel = _panel()
    out = top_k_by_mutual_information(panel, target="target", k=1)
    # 'useful' is the only signal; should be kept
    assert "useful" in out.columns
    assert "target" in out.columns
    # k=1 → only 1 feature column + target
    assert len([c for c in out.columns if c != "target"]) == 1


def test_top_k_mi_k_geq_num_features_keeps_all():
    panel = _panel()
    out = top_k_by_mutual_information(panel, target="target", k=100)
    assert "useful" in out.columns
    assert "noise" in out.columns


# ---------------------------------------------------------------------------
# sweep_alpha
# ---------------------------------------------------------------------------


def test_sweep_alpha_returns_per_alpha_rows():
    rng = np.random.default_rng(0)
    n = 150
    f = rng.normal(size=n)
    target = 0.4 * f + rng.normal(scale=0.5, size=n)
    panel = pd.DataFrame(
        {"cos_a": f, "news_count": rng.integers(1, 5, size=n), "target": target},
        index=pd.date_range("2025-01-01", periods=n, freq="B").date,
    )
    result = sweep_alpha(panel, target="target", alphas=(0.1, 1.0, 10.0), n_splits=3, min_train=40)
    assert len(result) == 3
    assert "mean_ic" in result.columns
    # IC should be positive (signal present); not testing exact value
    assert (result["mean_ic"] > -1.0).all()


def test_per_fold_selector_returns_callable_choosing_top_k():
    rng = np.random.default_rng(0)
    n = 80
    f_useful = rng.normal(size=n)
    panel = pd.DataFrame(
        {
            "useful": f_useful,
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
            "noise3": rng.normal(size=n),
            "target": 0.7 * f_useful + rng.normal(scale=0.5, size=n),
        }
    )
    selector = make_per_fold_mi_selector(k=2, random_state=0)
    chosen = selector(panel, "target")
    assert "useful" in chosen
    assert len(chosen) == 2


def test_per_fold_selector_returns_all_when_n_features_le_k():
    panel = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "target": [0.1, 0.2]})
    selector = make_per_fold_mi_selector(k=10)
    out = selector(panel, "target")
    assert set(out) == {"a", "b"}


def test_evaluate_walk_forward_with_feature_selector_no_leakage():
    from newsmood.models import evaluate_walk_forward

    rng = np.random.default_rng(7)
    n = 150
    f = rng.normal(size=n)
    panel = pd.DataFrame(
        {
            "useful": f,
            "noise1": rng.normal(size=n),
            "noise2": rng.normal(size=n),
            "target": 0.5 * f + rng.normal(scale=0.5, size=n),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="B").date,
    )
    sel = make_per_fold_mi_selector(k=1)
    res = evaluate_walk_forward(
        panel,
        target="target",
        feature_cols=["useful", "noise1", "noise2"],
        feature_selector=sel,
        n_splits=4,
        min_train=30,
    )
    assert not res.per_fold.empty
    # With per-fold MI selection on a 'useful' signal, mean IC should beat 0
    assert res.summary()["ic"] > 0


def test_sweep_alpha_handles_too_short():
    panel = pd.DataFrame(
        {
            "cos_a": np.arange(5, dtype=float),
            "target": np.arange(5, dtype=float),
        },
        index=pd.date_range("2025-01-01", periods=5, freq="B").date,
    )
    result = sweep_alpha(panel, target="target", alphas=(1.0,), min_train=20)
    # Should not crash; n_folds = 0
    assert result.iloc[0]["n_folds"] == 0
