"""Tests for newsmood.models."""

import numpy as np
import pandas as pd
import pytest

from newsmood.models import (
    evaluate_walk_forward,
    fit_predict_ridge,
    information_coefficient,
    long_short_sharpe,
    rank_information_coefficient,
    sign_accuracy,
    walk_forward_splits,
)


# -- splits ------------------------------------------------------------------


def test_walk_forward_basic_shapes():
    splits = list(walk_forward_splits(20, n_splits=3, min_train=8, embargo=1, test_size=3))
    assert len(splits) == 3
    for tr, te in splits:
        # No overlap, at least one row gap
        assert tr.max() + 1 < te.min()


def test_walk_forward_expanding_train():
    splits = list(walk_forward_splits(20, n_splits=3, min_train=8, embargo=1, test_size=3))
    sizes = [len(tr) for tr, _ in splits]
    assert sizes == sorted(sizes)  # non-decreasing


def test_walk_forward_short_data_returns_empty():
    assert list(walk_forward_splits(5, n_splits=3, min_train=10)) == []


# -- metrics -----------------------------------------------------------------


def test_information_coefficient_perfect():
    ic = information_coefficient([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert ic == pytest.approx(1.0)


def test_information_coefficient_handles_nans():
    # NaN row at index 2 is dropped; remaining 3 values are perfectly correlated.
    ic = information_coefficient([1.0, 2.0, float("nan"), 3.0], [1.0, 2.0, 99.0, 3.0])
    assert ic == pytest.approx(1.0)


def test_information_coefficient_insufficient():
    assert np.isnan(information_coefficient([1.0], [1.0]))


def test_rank_ic_inverse():
    assert rank_information_coefficient([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_sign_accuracy_basic():
    assert sign_accuracy([1.0, -1.0, 1.0], [0.5, -0.5, 0.1]) == pytest.approx(1.0)


def test_sign_accuracy_skips_zero_true():
    # The middle truly-zero row is excluded
    assert sign_accuracy([1.0, 0.0, -1.0], [1.0, 1.0, -1.0]) == pytest.approx(1.0)


def test_long_short_sharpe_positive():
    # Predictions perfectly match sign of returns
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=100)
    sharpe = long_short_sharpe(rets, rets)
    assert sharpe > 0


def test_long_short_sharpe_with_costs_lower():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=100)
    s0 = long_short_sharpe(rets, rets, cost_bps=0)
    s1 = long_short_sharpe(rets, rets, cost_bps=10)
    assert s1 < s0


# -- ridge fit ---------------------------------------------------------------


def test_fit_predict_ridge_recovers_linear():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    beta = np.array([1.0, -0.5, 2.0])
    y = X @ beta + rng.normal(scale=0.01, size=200)
    pred = fit_predict_ridge(X[:150], y[:150], X[150:], alpha=0.01)
    # Strong correlation with the truth
    assert np.corrcoef(pred, y[150:])[0, 1] > 0.99


# -- evaluate_walk_forward end-to-end ---------------------------------------


def _make_synthetic_panel(n_sessions=120, signal_strength=0.4, seed=0):
    rng = np.random.default_rng(seed)
    sessions = pd.date_range("2024-01-01", periods=n_sessions, freq="B").date
    f1 = rng.normal(size=n_sessions)
    f2 = rng.normal(size=n_sessions)
    noise = rng.normal(scale=1.0, size=n_sessions)
    y = signal_strength * f1 - 0.2 * f2 + noise
    return pd.DataFrame(
        {"cos_a": f1, "cos_b": f2, "news_count": rng.integers(1, 10, size=n_sessions), "target": y},
        index=pd.Index(sessions, name="session"),
    )


def test_evaluate_walk_forward_recovers_signal():
    panel = _make_synthetic_panel(n_sessions=200, signal_strength=0.6, seed=42)
    res = evaluate_walk_forward(
        panel,
        target="target",
        feature_cols=["cos_a", "cos_b", "news_count"],
        n_splits=4,
        min_train=60,
        embargo=1,
    )
    assert not res.per_fold.empty
    # Out-of-sample IC should be positive on this synthetic data
    summary = res.summary()
    assert summary["ic"] > 0
    assert summary["rank_ic"] > 0


def test_evaluate_walk_forward_no_signal_near_zero():
    panel = _make_synthetic_panel(n_sessions=200, signal_strength=0.0, seed=1)
    res = evaluate_walk_forward(
        panel, target="target", feature_cols=["cos_a", "cos_b", "news_count"]
    )
    summary = res.summary()
    # No-signal case: |IC| should be small in magnitude
    assert abs(summary["ic"]) < 0.4


def test_evaluate_walk_forward_no_features_raises():
    panel = _make_synthetic_panel(n_sessions=50)
    panel = panel.rename(columns={"cos_a": "x", "cos_b": "y", "news_count": "z"})
    with pytest.raises(ValueError, match="No feature columns"):
        evaluate_walk_forward(panel, target="target")


def test_evaluate_walk_forward_empty():
    res = evaluate_walk_forward(pd.DataFrame(), target="target")
    assert res.per_fold.empty
