"""Tests for newsmood.backtest."""

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from newsmood.backtest import (
    BacktestResult,
    annualized_sharpe,
    annualized_sortino,
    backtest,
    backtest_walk_forward,
    hit_rate_sign,
    max_drawdown,
    positions_decile_long_short,
    positions_sign,
    positions_tanh,
    turnover,
    win_rate,
)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_positions_sign_basic():
    out = positions_sign(pd.Series([0.5, -0.2, 0.0, np.nan]))
    assert out.tolist() == [1.0, -1.0, 0.0, 0.0]


def test_positions_tanh_clamped():
    out = positions_tanh(pd.Series([10.0, -10.0, 0.0]), scale=0.01)
    assert all(-1.0 <= v <= 1.0 for v in out)
    assert out.iloc[0] > 0.99
    assert out.iloc[1] < -0.99


def test_positions_decile_long_short():
    out = positions_decile_long_short(pd.Series([1, 2, 3, 4, 5]), top_q=0.4)
    assert out.tolist() == [-1.0, -1.0, 0.0, 1.0, 1.0]


def test_positions_decile_empty():
    out = positions_decile_long_short(pd.Series([], dtype=float))
    assert out.empty


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def test_sharpe_zero_std_returns_nan():
    # Use integers so std is exactly 0 (avoids float-imprecision std~1e-19)
    pnl = pd.Series([0.0] * 100)
    assert math.isnan(annualized_sharpe(pnl))


def test_sharpe_with_some_noise():
    rng = np.random.default_rng(0)
    pnl = pd.Series(rng.normal(0.001, 0.01, size=300))
    s = annualized_sharpe(pnl)
    assert math.isfinite(s)


def test_sortino_only_downside_in_denominator():
    rng = np.random.default_rng(0)
    pnl = pd.Series(rng.normal(0.001, 0.01, size=300))
    sortino = annualized_sortino(pnl)
    sharpe = annualized_sharpe(pnl)
    # Sortino >= Sharpe (downside-only std is <= full std)
    assert sortino >= sharpe - 1e-6


def test_sortino_no_downside_returns_inf_or_zero():
    pnl = pd.Series([0.01, 0.02, 0.03, 0.005, 0.04])  # never negative
    result = annualized_sortino(pnl)
    assert result == float("inf") or result == 0.0


def test_max_drawdown_known():
    e = pd.Series([1.0, 1.5, 1.2, 0.9, 1.3])
    assert max_drawdown(e) == pytest.approx(-0.4)


def test_max_drawdown_monotone_up_is_zero():
    e = pd.Series([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(e) == 0.0


def test_win_rate_excludes_zeros():
    pnl = pd.Series([0.01, -0.01, 0.0, 0.02, 0.0])
    # 2 wins, 1 loss, 2 zeros -> win_rate = 2/3
    assert win_rate(pnl) == pytest.approx(2 / 3)


def test_turnover_basic():
    positions = pd.Series([0.0, 0.5, -0.5, -0.5])
    # diffs: 0.5, 1.0, 0.0 -> mean = 0.5
    assert turnover(positions) == pytest.approx(0.5)


def test_hit_rate_sign():
    y_true = pd.Series([0.01, -0.01, 0.0, -0.02])
    y_pred = pd.Series([0.005, -0.005, 0.5, 0.01])
    # rows 0 (+/+), 1 (-/-), 3 (-/+) — zero-true skipped
    # 2 of 3 correct
    assert hit_rate_sign(y_true, y_pred) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------


def _aligned_series(n: int = 100, signal_strength: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    sessions = pd.date_range("2025-01-01", periods=n, freq="B").date
    returns = rng.normal(0.0005, 0.012, size=n)
    noise = rng.normal(scale=0.5, size=n)
    preds = signal_strength * returns + noise * 0.01
    return (
        pd.Series(returns, index=sessions, name="y_true"),
        pd.Series(preds, index=sessions, name="y_pred"),
    )


def test_backtest_shape_and_stats():
    y_true, y_pred = _aligned_series(120, signal_strength=2.0, seed=42)
    res = backtest(y_true, y_pred, sizing="tanh")
    assert isinstance(res, BacktestResult)
    # Equity series length matches input
    assert len(res.equity) == 120
    # First and last equity are reasonable
    assert res.equity.iloc[0] != 0
    # Stats keys present
    for k in ("sharpe", "sortino", "max_drawdown", "win_rate", "turnover", "n_periods", "final_equity"):
        assert k in res.stats


def test_backtest_strong_signal_positive_sharpe():
    y_true, y_pred = _aligned_series(300, signal_strength=10.0, seed=7)
    res = backtest(y_true, y_pred, sizing="sign", cost_bps=0.0)
    assert res.stats["sharpe"] > 0
    assert res.stats["final_equity"] > 1.0


def test_backtest_costs_lower_sharpe():
    y_true, y_pred = _aligned_series(300, signal_strength=5.0, seed=1)
    res_no_cost = backtest(y_true, y_pred, sizing="sign", cost_bps=0.0)
    res_high_cost = backtest(y_true, y_pred, sizing="sign", cost_bps=50.0)
    assert res_high_cost.stats["sharpe"] < res_no_cost.stats["sharpe"]


def test_backtest_costs_lower_final_equity():
    y_true, y_pred = _aligned_series(200, signal_strength=5.0, seed=2)
    res_no_cost = backtest(y_true, y_pred, sizing="sign", cost_bps=0.0)
    res_high_cost = backtest(y_true, y_pred, sizing="sign", cost_bps=50.0)
    assert res_high_cost.stats["final_equity"] <= res_no_cost.stats["final_equity"]


def test_backtest_no_signal_near_zero_sharpe():
    rng = np.random.default_rng(0)
    n = 500
    sessions = pd.date_range("2025-01-01", periods=n, freq="B").date
    y_true = pd.Series(rng.normal(0.0005, 0.012, size=n), index=sessions)
    y_pred = pd.Series(rng.normal(0.0, 1.0, size=n), index=sessions)
    res = backtest(y_true, y_pred, sizing="sign", cost_bps=0.0)
    assert abs(res.stats["sharpe"]) < 3.0  # not crazy


def test_backtest_equity_drops_below_one_for_bad_signal():
    """An inverted signal applied with sign+no cost should lose money."""
    y_true, y_pred = _aligned_series(300, signal_strength=10.0, seed=99)
    res = backtest(y_true, -y_pred, sizing="sign", cost_bps=0.0)
    assert res.stats["final_equity"] < 1.0
    assert res.stats["sharpe"] < 0


def test_backtest_zero_position_pnl():
    # All-zero predictions → zero positions → zero PnL
    n = 50
    sessions = pd.date_range("2025-01-01", periods=n, freq="B").date
    y_true = pd.Series(np.linspace(-0.01, 0.01, n), index=sessions)
    y_pred = pd.Series(np.zeros(n), index=sessions)
    res = backtest(y_true, y_pred, sizing="sign", cost_bps=0.0)
    assert res.stats["final_equity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# backtest_walk_forward integration with newsmood.models
# ---------------------------------------------------------------------------


def test_backtest_walk_forward_from_wf_results():
    from newsmood.models import evaluate_walk_forward

    # Build a panel where preds should track target tightly
    rng = np.random.default_rng(0)
    sessions = pd.date_range("2025-01-01", periods=120, freq="B").date
    f = rng.normal(size=120)
    target = 0.8 * f + 0.2 * rng.normal(size=120)
    panel = pd.DataFrame(
        {"cos_a": f, "news_count": rng.integers(1, 5, size=120), "target": target},
        index=pd.Index(sessions, name="session"),
    )
    res = evaluate_walk_forward(
        panel, target="target", feature_cols=["cos_a", "news_count"],
        n_splits=4, min_train=30, embargo=1,
    )
    bt = backtest_walk_forward(res.predictions, sizing="tanh")
    assert isinstance(bt, BacktestResult)
    assert bt.stats["n_periods"] > 0
    # With strong synthetic signal, Sharpe should be positive
    assert bt.stats["sharpe"] > 0
