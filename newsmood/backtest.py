"""Backtest a per-session prediction series into a tradable P&L.

The contract: given a Series of ``y_pred`` indexed by session date plus the
realized forward return ``y_true`` at the same horizon, produce an equity
curve and a battery of standard performance stats.

Trading convention
------------------
For each session ``t``:

- ``position_t = sign(y_pred_t) * weight``,
  where ``weight`` defaults to a continuous tanh-squashed conviction
  (``tanh(y_pred / scale)``), capped to ``[-1, 1]`` of NAV.
- Realised P&L for session ``t`` is ``position_t * y_true_t`` (already in
  log-return space).
- Transaction cost: ``cost_bps`` of NAV is charged on the change in position
  ``|position_t - position_{t-1}|``.

The numbers below are *strategy* metrics, not portfolio metrics — no leverage
budget, no slippage model. They're the canonical first-pass scoring of a
single signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def positions_sign(y_pred: pd.Series) -> pd.Series:
    """Discrete ±1 / 0 positions.

    >>> import pandas as pd
    >>> positions_sign(pd.Series([0.5, -0.2, 0.0])).tolist()
    [1.0, -1.0, 0.0]
    """
    return np.sign(y_pred.astype(float)).fillna(0.0)


def positions_tanh(y_pred: pd.Series, *, scale: float = 0.01) -> pd.Series:
    """Continuous ``tanh(y_pred / scale)`` sizing — clamped to ``[-1, 1]``.

    Default ``scale=0.01`` means a +1% predicted return → ≈0.76 position.

    >>> import pandas as pd, math
    >>> p = float(positions_tanh(pd.Series([0.01]), scale=0.01).iloc[0])
    >>> 0.76 < p < 0.77
    True
    """
    return np.tanh(y_pred.astype(float) / max(scale, 1e-12)).fillna(0.0)


def positions_decile_long_short(y_pred: pd.Series, *, top_q: float = 0.2) -> pd.Series:
    """Long top quantile, short bottom quantile, flat in the middle.

    For cross-sectional use; in single-asset pure-time-series mode the
    quantile threshold is computed over the **whole** series, so look-ahead
    is present — callers should compute per-fold thresholds for OOS reuse.

    >>> import pandas as pd
    >>> positions_decile_long_short(pd.Series([1, 2, 3, 4, 5]), top_q=0.4).tolist()
    [-1.0, -1.0, 0.0, 1.0, 1.0]
    """
    if y_pred.empty:
        return y_pred.copy()
    s = y_pred.astype(float)
    lo = s.quantile(top_q)
    hi = s.quantile(1.0 - top_q)
    out = pd.Series(0.0, index=s.index)
    out[s >= hi] = 1.0
    out[s <= lo] = -1.0
    return out


SIZING_FUNCS = {
    "sign": positions_sign,
    "tanh": positions_tanh,
    "decile": positions_decile_long_short,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def annualized_sharpe(
    pnl: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Sharpe ratio = mean / std × √(periods_per_year). NaN if too few samples."""
    p = pnl.dropna()
    if len(p) < 5:
        return float("nan")
    std = p.std(ddof=1)
    if std == 0 or not math.isfinite(std):
        return float("nan")
    return float((p.mean() / std) * math.sqrt(periods_per_year))


def annualized_sortino(
    pnl: pd.Series, *, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Sortino = mean / downside-std × √(periods_per_year)."""
    p = pnl.dropna()
    if len(p) < 5:
        return float("nan")
    downside = p[p < 0]
    if downside.empty:
        return float("inf") if p.mean() > 0 else 0.0
    std_d = downside.std(ddof=1)
    if std_d == 0 or not math.isfinite(std_d):
        return float("nan")
    return float((p.mean() / std_d) * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough drawdown as a fraction (negative number).

    >>> import pandas as pd
    >>> e = pd.Series([1.0, 1.5, 1.2, 0.9, 1.3])
    >>> round(max_drawdown(e), 4)
    -0.4
    """
    e = equity.dropna()
    if e.empty:
        return float("nan")
    peak = e.cummax()
    dd = e / peak - 1.0
    return float(dd.min())


def win_rate(pnl: pd.Series) -> float:
    """Fraction of strictly-positive P&L periods (zeros excluded)."""
    p = pnl.dropna()
    nonzero = p[p != 0]
    if nonzero.empty:
        return float("nan")
    return float((nonzero > 0).mean())


def turnover(positions: pd.Series) -> float:
    """Average |Δposition| per period — proxy for trading activity."""
    p = positions.dropna()
    if len(p) < 2:
        return 0.0
    deltas = p.diff().abs().dropna()
    return float(deltas.mean()) if not deltas.empty else 0.0


def hit_rate_sign(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of rows where ``sign(y_pred) == sign(y_true)`` (zeros skipped)."""
    a = y_true.astype(float)
    b = y_pred.astype(float)
    mask = a.notna() & b.notna() & (a != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(a[mask]) == np.sign(b[mask])).mean())


# ---------------------------------------------------------------------------
# Core backtest
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    equity: pd.Series  # cumulative equity (starts at 1.0)
    pnl: pd.Series  # per-period P&L after costs
    positions: pd.Series  # per-period position (signed weight)
    cost: pd.Series  # per-period transaction cost
    stats: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        keys = ("ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "win_rate", "hit_rate", "turnover", "n_periods")
        bits = []
        for k in keys:
            v = self.stats.get(k)
            if v is None:
                continue
            if isinstance(v, float):
                bits.append(f"{k}={v:.4f}")
            else:
                bits.append(f"{k}={v}")
        return "BacktestResult(" + ", ".join(bits) + ")"

    def summary(self) -> pd.Series:
        return pd.Series(self.stats)


def backtest(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    sizing: str = "tanh",
    sizing_kwargs: Optional[dict] = None,
    cost_bps: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestResult:
    """Run a single-asset, single-signal backtest.

    Parameters
    ----------
    y_true, y_pred
        Aligned Series of realized forward returns and model predictions
        (same index ordering).
    sizing
        Name of a sizing function in :data:`SIZING_FUNCS` (``"sign"``,
        ``"tanh"``, ``"decile"``) or a custom callable
        ``(pd.Series) -> pd.Series``.
    sizing_kwargs
        Forwarded to the sizing function (e.g. ``{"scale": 0.005}`` for
        ``tanh``).
    cost_bps
        Per-period transaction cost charged on |Δposition| in basis points
        of NAV. ``1.0`` = 1bp = 0.01%. Default 1 bp is realistic for SPY-
        like ETFs at retail brokers.
    periods_per_year
        For annualization.
    """
    if isinstance(sizing, str):
        size_fn = SIZING_FUNCS[sizing]
    else:
        size_fn = sizing
    positions = size_fn(y_pred, **(sizing_kwargs or {})).reindex(y_pred.index).fillna(0.0)

    # Δposition: first-period takes on its position from 0.
    delta = positions.diff().fillna(positions.iloc[0] if not positions.empty else 0.0).abs()
    cost = delta * (cost_bps / 1e4)
    gross_pnl = (positions * y_true).fillna(0.0)
    pnl = gross_pnl - cost

    equity = (1.0 + pnl).cumprod()

    n = len(pnl)
    ann_return = float((1 + pnl.mean()) ** periods_per_year - 1) if n else float("nan")
    ann_vol = float(pnl.std(ddof=1) * math.sqrt(periods_per_year)) if n >= 2 else float("nan")
    stats = {
        "n_periods": n,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": annualized_sharpe(pnl, periods_per_year=periods_per_year),
        "sortino": annualized_sortino(pnl, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(pnl),
        "hit_rate": hit_rate_sign(y_true, y_pred),
        "turnover": turnover(positions),
        "avg_pos_abs": float(positions.abs().mean()) if n else float("nan"),
        "total_cost": float(cost.sum()) if n else 0.0,
        "final_equity": float(equity.iloc[-1]) if n else 1.0,
    }
    return BacktestResult(equity=equity, pnl=pnl, positions=positions, cost=cost, stats=stats)


def backtest_walk_forward(
    walk_forward_predictions: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    sizing: str = "tanh",
    sizing_kwargs: Optional[dict] = None,
    cost_bps: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestResult:
    """Backtest from the long-form predictions DataFrame produced by
    :func:`newsmood.models.evaluate_walk_forward`.

    The DataFrame must have a session-date index and the columns
    ``y_true``/``y_pred``. Concatenates all folds; cost is recomputed on the
    full series (so fold boundaries pay turnover only if positions actually
    flip there).
    """
    df = walk_forward_predictions.sort_index()
    return backtest(
        df[y_true_col],
        df[y_pred_col],
        sizing=sizing,
        sizing_kwargs=sizing_kwargs,
        cost_bps=cost_bps,
        periods_per_year=periods_per_year,
    )


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "SIZING_FUNCS",
    "positions_sign",
    "positions_tanh",
    "positions_decile_long_short",
    "annualized_sharpe",
    "annualized_sortino",
    "max_drawdown",
    "win_rate",
    "turnover",
    "hit_rate_sign",
    "BacktestResult",
    "backtest",
    "backtest_walk_forward",
]
