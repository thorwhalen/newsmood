"""Feature selection + regularization tuning for the news panel.

These helpers are small and orthogonal — they can be composed in any order
before handing the result to :func:`newsmood.models.evaluate_walk_forward`.

- :func:`drop_low_variance`: remove features whose stdev is below a cutoff
  (typically near-constant ``q_*`` columns from a thin news category).
- :func:`drop_high_correlation`: when two features have ``|corr| > thresh``,
  drop the one with the lower correlation to the target — reduces collinearity.
- :func:`top_k_by_mutual_information`: keep the K features with highest MI
  against the target. Robust to non-linear relationships.
- :func:`sweep_alpha`: walk-forward Ridge sweep over a grid of ``alpha`` values,
  returning a DataFrame of per-fold and mean metrics for each alpha.

Examples
--------
>>> import pandas as pd, numpy as np
>>> rng = np.random.default_rng(0)
>>> panel = pd.DataFrame({
...     'a': rng.normal(size=50),
...     'b': np.zeros(50),  # zero-variance
...     'target': rng.normal(size=50),
... })
>>> reduced = drop_low_variance(panel, target='target')
>>> 'b' in reduced.columns
False
>>> 'a' in reduced.columns
True
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Variance filter
# ---------------------------------------------------------------------------


def drop_low_variance(
    panel: pd.DataFrame,
    *,
    target: Optional[str] = None,
    min_std: float = 1e-6,
    feature_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Drop features whose stdev is below ``min_std``.

    The target column is preserved. If ``feature_cols`` is None, every column
    other than ``target`` is treated as a feature.
    """
    if feature_cols is None:
        feature_cols = [c for c in panel.columns if c != target]
    feature_cols_set = set(feature_cols)
    drop = {c for c in feature_cols_set if panel[c].std(ddof=0) <= min_std}
    keep = [c for c in panel.columns if c not in drop]
    return panel[keep].copy()


# ---------------------------------------------------------------------------
# Correlation filter
# ---------------------------------------------------------------------------


def drop_high_correlation(
    panel: pd.DataFrame,
    *,
    target: str,
    threshold: float = 0.95,
    feature_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Drop one of each pair of features with ``|corr| > threshold``.

    The dropped column is the one with the **smaller** absolute Pearson
    correlation to the target, so we preserve the more target-relevant of
    each redundant pair.
    """
    if feature_cols is None:
        feature_cols = [c for c in panel.columns if c != target]
    feature_cols = list(feature_cols)
    work = panel[feature_cols + [target]].dropna()
    if work.empty or len(feature_cols) <= 1:
        return panel.copy()

    feat_corr = work[feature_cols].corr().abs()
    tgt_corr = work[feature_cols].corrwith(work[target]).abs()

    to_drop: set[str] = set()
    for i, a in enumerate(feature_cols):
        if a in to_drop:
            continue
        for b in feature_cols[i + 1 :]:
            if b in to_drop:
                continue
            if feat_corr.loc[a, b] > threshold:
                drop = a if tgt_corr.get(a, 0) < tgt_corr.get(b, 0) else b
                to_drop.add(drop)

    keep = [c for c in panel.columns if c not in to_drop]
    return panel[keep].copy()


# ---------------------------------------------------------------------------
# Mutual-information filter
# ---------------------------------------------------------------------------


def top_k_by_mutual_information(
    panel: pd.DataFrame,
    *,
    target: str,
    k: int = 20,
    feature_cols: Optional[Iterable[str]] = None,
    random_state: Optional[int] = 0,
) -> pd.DataFrame:
    """Keep the ``k`` features with highest mutual information against the target.

    Uses ``sklearn.feature_selection.mutual_info_regression``. Robust to
    monotonic non-linearities; ties broken by original column order.
    """
    from sklearn.feature_selection import mutual_info_regression

    if feature_cols is None:
        feature_cols = [c for c in panel.columns if c != target]
    feature_cols = list(feature_cols)
    work = panel[feature_cols + [target]].dropna()
    if work.empty:
        return panel.copy()

    X = work[feature_cols].to_numpy(dtype=float)
    y = work[target].to_numpy(dtype=float)
    mi = mutual_info_regression(X, y, random_state=random_state)
    mi_series = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
    top = mi_series.head(k).index.tolist()
    keep = top + [target]
    return panel[[c for c in panel.columns if c in keep]].copy()


# ---------------------------------------------------------------------------
# Alpha sweep
# ---------------------------------------------------------------------------


def sweep_alpha(
    panel: pd.DataFrame,
    *,
    target: str,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0),
    feature_cols: Optional[Iterable[str]] = None,
    n_splits: int = 5,
    min_train: int = 30,
    embargo: int = 5,
) -> pd.DataFrame:
    """Walk-forward Ridge sweep over a grid of ``alpha`` values.

    Returns a DataFrame indexed by alpha with mean and std of IC, rank-IC,
    sign accuracy, and Sharpe across folds.
    """
    from newsmood.models import evaluate_walk_forward

    rows: list[dict] = []
    for alpha in alphas:
        res = evaluate_walk_forward(
            panel,
            target=target,
            feature_cols=feature_cols,
            n_splits=n_splits,
            min_train=min_train,
            embargo=embargo,
            alpha=alpha,
        )
        if res.per_fold.empty:
            rows.append({"alpha": alpha, "n_folds": 0})
            continue
        rows.append(
            {
                "alpha": alpha,
                "n_folds": len(res.per_fold),
                "mean_ic": float(res.per_fold["ic"].mean()),
                "std_ic": float(res.per_fold["ic"].std(ddof=1)),
                "mean_rank_ic": float(res.per_fold["rank_ic"].mean()),
                "mean_sign_acc": float(res.per_fold["sign_acc"].mean()),
                "mean_sharpe": float(res.per_fold["sharpe"].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("alpha").sort_index()


def make_per_fold_mi_selector(
    k: int = 20, *, random_state: Optional[int] = 0
):
    """Return a closure ``(train_df, target) -> [columns]`` usable as
    ``evaluate_walk_forward(..., feature_selector=...)``.

    Per-fold MI selection — runs only on the **train slice**, so OOS folds
    can't leak through feature choice. This is the leakage-free counterpart
    to :func:`top_k_by_mutual_information` (which is global).
    """
    from sklearn.feature_selection import mutual_info_regression

    def selector(train_df: pd.DataFrame, target: str) -> list[str]:
        feature_cols = [c for c in train_df.columns if c != target]
        work = train_df[feature_cols + [target]].dropna()
        if work.empty or len(feature_cols) <= k:
            return feature_cols
        X = work[feature_cols].to_numpy(dtype=float)
        y = work[target].to_numpy(dtype=float)
        mi = mutual_info_regression(X, y, random_state=random_state)
        s = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
        return s.head(k).index.tolist()

    return selector


__all__ = [
    "drop_low_variance",
    "drop_high_correlation",
    "make_per_fold_mi_selector",
    "top_k_by_mutual_information",
    "sweep_alpha",
]
