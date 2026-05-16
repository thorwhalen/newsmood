"""Baseline walk-forward modeling on the news-feature panel.

A deliberately small surface aimed at the first vertical slice:

- :func:`walk_forward_splits`: expanding-window splits with an embargo, so
  no test fold can leak information from training (forward-return overlap).
- :func:`information_coefficient` / :func:`rank_information_coefficient`:
  the two metrics the user's :doc:`hedger/RESEARCH.md` cites as the canonical
  factor-research scoring methods.
- :func:`fit_predict_ridge`: single fold fit-and-predict.
- :func:`evaluate_walk_forward`: glues the two together; returns per-fold
  metrics + a long-form DataFrame of out-of-sample predictions.

Out of scope here (deferred): hyperparameter tuning, classification head,
panel models with ticker fixed effects, RL bolt-on. The current code is
intentionally a baseline — it answers "is there any usable signal?".
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd


# -- Splits ------------------------------------------------------------------


def walk_forward_splits(
    n: int,
    *,
    n_splits: int = 5,
    min_train: int = 30,
    embargo: int = 1,
    test_size: Optional[int] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` pairs for expanding-window CV.

    Each test fold begins ``embargo`` indices *after* the last train index, so
    the forward-return horizon — which makes label ``r_t`` peek at price
    ``t + h`` — cannot leak across the train/test boundary as long as
    ``embargo >= h``.

    Parameters
    ----------
    n
        Total number of (already chronologically-ordered) rows.
    n_splits
        How many folds to produce.
    min_train
        Minimum number of rows in the first training fold.
    embargo
        Gap (in rows / sessions) between train and test.
    test_size
        Size of each test fold. Default: ``(n - min_train) // n_splits``.

    >>> list(walk_forward_splits(20, n_splits=3, min_train=8, embargo=1, test_size=3))
    [(array([0, 1, 2, 3, 4, 5, 6, 7]), array([ 9, 10, 11])), (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]), array([13, 14, 15])), (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]), array([17, 18, 19]))]
    """
    if n < min_train + embargo + 1:
        return
    if test_size is None:
        test_size = max(1, (n - min_train) // n_splits)
    cur = min_train
    for _ in range(n_splits):
        train_end = cur
        test_start = train_end + embargo
        test_end = min(n, test_start + test_size)
        if test_start >= n:
            return
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx
        cur += test_size
        if cur + embargo >= n:
            return


# -- Metrics -----------------------------------------------------------------


def information_coefficient(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> float:
    """Pearson IC. ``NaN``-tolerant; returns ``nan`` if insufficient data.

    >>> round(information_coefficient([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 6)
    1.0
    """
    a = pd.Series(y_true)
    b = pd.Series(y_pred)
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(a[mask].corr(b[mask], method="pearson"))


def rank_information_coefficient(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> float:
    """Spearman IC. ``NaN``-tolerant.

    >>> round(rank_information_coefficient([1, 2, 3], [3, 2, 1]), 6)
    -1.0
    """
    a = pd.Series(y_true)
    b = pd.Series(y_pred)
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan")
    return float(a[mask].corr(b[mask], method="spearman"))


def sign_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Fraction of predictions whose sign matches the true sign. Zero-true rows skipped.

    >>> sign_accuracy([1, -1, 1], [0.5, -0.2, 0.3])
    1.0
    """
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (a != 0)
    if not mask.any():
        return float("nan")
    return float((np.sign(a[mask]) == np.sign(b[mask])).mean())


def long_short_sharpe(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    cost_bps: float = 0.0,
) -> float:
    """Sharpe of a sign-based long-short portfolio.

    For each session, go long if ``y_pred > 0``, short if ``y_pred < 0``, scale
    by 1; realized return is ``sign(y_pred) * y_true``. Optional per-trade
    transaction cost in basis points is subtracted when the sign flips.
    """
    a = pd.Series(y_true, dtype=float)
    b = pd.Series(y_pred, dtype=float)
    mask = a.notna() & b.notna()
    if mask.sum() < 5:
        return float("nan")
    a = a[mask].reset_index(drop=True)
    b = b[mask].reset_index(drop=True)
    sig = np.sign(b)
    pnl = sig * a
    if cost_bps > 0:
        flips = (sig != sig.shift(1)).astype(int)
        flips.iloc[0] = 1  # entering the first position costs too
        pnl = pnl - flips * (cost_bps / 1e4)
    std = pnl.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return float("nan")
    # Annualize assuming 252 trading sessions
    return float((pnl.mean() / std) * np.sqrt(252))


# -- Fit / predict -----------------------------------------------------------


def fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Standard-scale + Ridge fit, return test predictions."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe.predict(X_test)


# -- Evaluation glue ---------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Output container for :func:`evaluate_walk_forward`."""

    target: str
    per_fold: pd.DataFrame  # one row per fold with metrics
    predictions: pd.DataFrame  # long-form: index | y_true | y_pred | fold
    feature_names: list[str] = field(default_factory=list)

    def summary(self) -> pd.Series:
        """Aggregate metrics across folds (mean)."""
        cols = [c for c in self.per_fold.columns if c not in ("fold", "n_train", "n_test")]
        return self.per_fold[cols].mean().rename("mean")


def evaluate_walk_forward(
    panel: pd.DataFrame,
    *,
    target: str,
    feature_cols: Optional[Iterable[str]] = None,
    n_splits: int = 5,
    min_train: int = 30,
    embargo: int = 1,
    alpha: float = 1.0,
    cost_bps: float = 0.0,
    feature_selector: Optional[Callable[[pd.DataFrame, str], list[str]]] = None,
) -> WalkForwardResult:
    """Walk-forward Ridge baseline over a session-level panel.

    The panel must be sorted by session ascending. Rows with NaN target or
    feature values are dropped before training.

    Parameters
    ----------
    feature_selector
        Optional callable ``(train_df, target_name) -> list[selected_features]``.
        Called *inside each fold* on the **train slice only** — this avoids the
        look-ahead leakage that occurs when feature selection is run globally
        before splitting. The selector chooses among ``feature_cols``.
    """
    if panel.empty:
        return WalkForwardResult(target=target, per_fold=pd.DataFrame(), predictions=pd.DataFrame())

    if feature_cols is None:
        feature_cols = [
            c
            for c in panel.columns
            if c == "news_count"
            or c.startswith("cos_")
            or c.startswith("q_")
            or c.startswith("mean_vec_")
        ]
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("No feature columns matched defaults; specify feature_cols.")

    work = panel.dropna(subset=[target] + feature_cols).copy().sort_index()
    n = len(work)
    if n < min_train + embargo + 1:
        return WalkForwardResult(target=target, per_fold=pd.DataFrame(), predictions=pd.DataFrame())

    fold_metrics: list[dict] = []
    pred_records: list[pd.DataFrame] = []

    for fold, (tr, te) in enumerate(
        walk_forward_splits(n, n_splits=n_splits, min_train=min_train, embargo=embargo)
    ):
        # Optional per-fold feature selection on train-only data (no leakage).
        if feature_selector is None:
            cols_this_fold = feature_cols
        else:
            train_df = work.iloc[tr]
            cols_this_fold = list(feature_selector(train_df, target))
            if not cols_this_fold:
                cols_this_fold = feature_cols  # fall back

        X = work[cols_this_fold].to_numpy(dtype=float)
        y = work[target].to_numpy(dtype=float)
        y_pred = fit_predict_ridge(X[tr], y[tr], X[te], alpha=alpha)
        y_test = y[te]
        fold_metrics.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "ic": information_coefficient(y_test, y_pred),
                "rank_ic": rank_information_coefficient(y_test, y_pred),
                "sign_acc": sign_accuracy(y_test, y_pred),
                "sharpe": long_short_sharpe(y_test, y_pred, cost_bps=cost_bps),
            }
        )
        pred_records.append(
            pd.DataFrame(
                {
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "fold": fold,
                },
                index=work.index[te],
            )
        )

    per_fold = pd.DataFrame(fold_metrics)
    predictions = (
        pd.concat(pred_records) if pred_records else pd.DataFrame(columns=["y_true", "y_pred", "fold"])
    )
    return WalkForwardResult(
        target=target,
        per_fold=per_fold,
        predictions=predictions,
        feature_names=feature_cols,
    )


__all__ = [
    "WalkForwardResult",
    "evaluate_walk_forward",
    "fit_predict_ridge",
    "information_coefficient",
    "long_short_sharpe",
    "rank_information_coefficient",
    "sign_accuracy",
    "walk_forward_splits",
]
