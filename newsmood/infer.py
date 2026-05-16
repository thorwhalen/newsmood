"""Train, persist, and apply a news-embedding signal model.

Two halves:

- :func:`train_and_save`: fit the baseline Ridge pipeline on a labeled panel,
  pickle the fitted pipeline plus the feature schema and a model version.
- :func:`load_model` / :func:`predict_scores` / :func:`write_predictions_to_mall`:
  consume the persisted model to score new feature frames, then write per-
  ``(symbol, session)`` scores into a mall the
  :mod:`hedger.strategies.news_embed` strategy reads from.

The mall is any ``MutableMapping`` — a dict for tests, a ``dol`` store for
production. The persisted feature_key is ``features:news_embed_v1`` by default
(matches the strategy's default).
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from newsmood.models import fit_predict_ridge


DEFAULT_MODEL_ROOT = os.environ.get(
    "NEWSMOOD_MODEL_ROOT",
    str(pathlib.Path("~/.config/newsmood/models").expanduser()),
)
DEFAULT_FEATURE_KEY = "features:news_embed_v1"


# ---------------------------------------------------------------------------
# Persisted-model container
# ---------------------------------------------------------------------------


@dataclass
class InferenceModel:
    """A fitted Ridge pipeline plus the schema needed to apply it.

    Attributes
    ----------
    pipeline
        A fitted sklearn ``Pipeline`` (typically ``StandardScaler → Ridge``).
    feature_cols
        Column names — both required and order-sensitive.
    target
        Target column the model was trained against (for provenance).
    version
        A free-form version string (e.g. ``"v1"``, ``"2026-05-15"``).
    metadata
        Anything else worth keeping: training row count, IC summary, etc.
    """

    pipeline: Any
    feature_cols: list[str]
    target: str
    version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict on a feature frame; missing feature columns raise."""
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"Feature columns missing from input: {missing[:5]}"
                + (f" (+ {len(missing) - 5} more)" if len(missing) > 5 else "")
            )
        X_use = X[self.feature_cols].copy()
        # Median-impute any NaNs (we already train on dropped-NaN rows, but
        # inference rows may have new NaNs from unseen sources).
        X_use = X_use.fillna(X_use.median(numeric_only=True))
        # Final safety: zero-fill columns that are entirely NaN.
        X_use = X_use.fillna(0.0)
        return self.pipeline.predict(X_use.to_numpy(dtype=float))


def train_and_save(
    panel: pd.DataFrame,
    *,
    target: str,
    feature_cols: Optional[Iterable[str]] = None,
    alpha: float = 1.0,
    model_path: Optional[str] = None,
    version: str = "v1",
    metadata: Optional[dict] = None,
) -> InferenceModel:
    """Fit on the full panel (no CV; CV is for evaluation) and persist.

    The CV pipeline lives in :func:`newsmood.models.evaluate_walk_forward` —
    that's where you decide *whether* the model has signal. Once you have
    chosen ``alpha`` / feature set there, run *this* function on the same
    panel to produce the operational artifact.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if feature_cols is None:
        feature_cols = [
            c
            for c in panel.columns
            if c == "news_count"
            or c.startswith("cos_")
            or c.startswith("q_")
            or c.startswith("clust_")
            or c.startswith("mean_vec_")
        ]
    feature_cols = list(feature_cols)
    if not feature_cols:
        raise ValueError("No feature columns matched defaults; specify feature_cols.")

    work = panel.dropna(subset=[target] + feature_cols)
    if work.empty:
        raise ValueError("No rows with both features and target are non-null.")
    X = work[feature_cols].to_numpy(dtype=float)
    y = work[target].to_numpy(dtype=float)

    pipe = Pipeline(
        [
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    pipe.fit(X, y)

    meta: dict[str, Any] = {
        "n_train": int(len(work)),
        "alpha": alpha,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        meta.update(metadata)

    model = InferenceModel(
        pipeline=pipe,
        feature_cols=feature_cols,
        target=target,
        version=version,
        metadata=meta,
    )
    if model_path is not None:
        save_model(model, model_path)
    return model


def save_model(model: InferenceModel, model_path: str) -> None:
    """Persist an InferenceModel to disk (pickle)."""
    p = pathlib.Path(model_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(model, f)


def load_model(model_path: str) -> InferenceModel:
    """Load an InferenceModel produced by :func:`save_model`."""
    p = pathlib.Path(model_path).expanduser()
    with p.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, InferenceModel):
        raise TypeError(f"File at {p} did not contain an InferenceModel")
    return obj


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_scores(
    model: InferenceModel, feature_df: pd.DataFrame
) -> pd.Series:
    """Apply model to feature_df; return Series indexed like feature_df."""
    preds = model.predict(feature_df)
    return pd.Series(preds, index=feature_df.index, name=f"score_{model.target}")


# ---------------------------------------------------------------------------
# Mall I/O
# ---------------------------------------------------------------------------


def _coerce_session(s: Any) -> date:
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, pd.Timestamp):
        return s.date()
    if isinstance(s, str):
        return datetime.fromisoformat(s).date()
    raise TypeError(f"Cannot coerce {s!r} to a session date")


def write_predictions_to_mall(
    predictions: pd.Series,
    *,
    mall: MutableMapping,
    ticker: Optional[str] = None,
    feature_key: str = DEFAULT_FEATURE_KEY,
) -> int:
    """Write per-(ticker, session) scores into ``mall[feature_key]``.

    Two index shapes are supported:

    - Single Index of session dates: ``ticker`` must be provided; every entry
      is keyed ``(ticker, session)``.
    - MultiIndex of ``(ticker, session)``: ``ticker`` is ignored.

    ``mall[feature_key]`` is created (as a plain ``dict``) if absent, or
    updated in place if present.

    Returns the number of entries written.
    """
    try:
        feature_store = mall[feature_key]
    except KeyError:
        # Create the entry, then re-fetch — some mall implementations
        # (e.g. a persistent dol-backed mall) wrap the value on insert, so
        # the local dict we just wrote isn't the same object as what the
        # mall now holds.
        mall[feature_key] = {}
        feature_store = mall[feature_key]

    if not isinstance(feature_store, MutableMapping):
        raise TypeError(
            f"mall[{feature_key!r}] must be a MutableMapping; got {type(feature_store)}"
        )

    written = 0
    if isinstance(predictions.index, pd.MultiIndex):
        # Expect (ticker, session) shape
        for (tkr, sess), score in predictions.items():
            if pd.isna(score):
                continue
            feature_store[(str(tkr), _coerce_session(sess))] = float(score)
            written += 1
    else:
        if ticker is None:
            raise ValueError(
                "predictions has a single index — pass ticker=... so we can build "
                "(ticker, session) keys."
            )
        for sess, score in predictions.items():
            if pd.isna(score):
                continue
            feature_store[(str(ticker), _coerce_session(sess))] = float(score)
            written += 1
    return written


def run_inference(
    feature_df: pd.DataFrame,
    *,
    model: InferenceModel,
    mall: MutableMapping,
    ticker: Optional[str] = None,
    feature_key: str = DEFAULT_FEATURE_KEY,
) -> int:
    """End-to-end: predict on ``feature_df`` and write to ``mall``.

    Returns the number of mall entries written.
    """
    preds = predict_scores(model, feature_df)
    return write_predictions_to_mall(
        preds, mall=mall, ticker=ticker, feature_key=feature_key
    )


# ---------------------------------------------------------------------------
# Mall convenience: dol-backed persistent dict-of-dicts
# ---------------------------------------------------------------------------


def open_mall(
    root: Optional[str] = None,
) -> MutableMapping:
    """Open a persistent mall — a dol-backed JSON store at ``root``.

    Each top-level key (e.g. ``features:news_embed_v1``) maps to a JSON-blob
    file at ``root``. Nested entries are JSON-coded ``[ticker, session-iso] -> float``
    pairs.

    For a quick in-process mall, just pass a ``dict()`` everywhere — this
    helper is for daemon-style daily inference jobs that need to persist.
    """
    if root is None:
        root = os.environ.get(
            "NEWSMOOD_MALL_ROOT",
            str(pathlib.Path("~/.config/newsmood/mall").expanduser()),
        )
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    return _JsonTupleKeyMall(root)


class _JsonTupleKeyMall(MutableMapping):
    """A minimal mall that JSON-codes tuple keys for persistence.

    Top-level keys (``feature_key``) map to sub-mappings; sub-mapping keys can
    be ``(ticker, session_date)`` or strings; sub-mapping values are floats.
    The on-disk format is one JSON file per top-level key. Sub-mappings are
    loaded eagerly and written back on each ``__setitem__`` to keep the API
    consistent with a regular dict.
    """

    def __init__(self, root: str):
        self._root = pathlib.Path(root)
        self._cache: dict[str, _PersistentSubMall] = {}

    def __getitem__(self, key: str) -> "_PersistentSubMall":
        if key in self._cache:
            return self._cache[key]
        path = self._root / f"{key.replace(':', '__')}.json"
        if not path.exists():
            raise KeyError(key)
        sub = _PersistentSubMall(path)
        self._cache[key] = sub
        return sub

    def __setitem__(self, key: str, value: Mapping) -> None:
        path = self._root / f"{key.replace(':', '__')}.json"
        sub = _PersistentSubMall(path)
        for k, v in value.items():
            sub[k] = v
        self._cache[key] = sub

    def __delitem__(self, key: str) -> None:
        path = self._root / f"{key.replace(':', '__')}.json"
        if path.exists():
            path.unlink()
        self._cache.pop(key, None)

    def __iter__(self):
        for p in self._root.glob("*.json"):
            yield p.stem.replace("__", ":")

    def __len__(self) -> int:
        return sum(1 for _ in self._root.glob("*.json"))


class _PersistentSubMall(MutableMapping):
    """A dict-shaped store whose entries persist to a JSON file."""

    def __init__(self, path: pathlib.Path):
        self._path = path
        self._data: dict[tuple, float] = {}
        if path.exists():
            try:
                with path.open() as f:
                    raw = json.load(f)
                for k_s, v in raw.items():
                    parts = k_s.split("|", 1)
                    if len(parts) == 2:
                        try:
                            sess = date.fromisoformat(parts[1])
                        except ValueError:
                            sess = parts[1]
                        self._data[(parts[0], sess)] = float(v)
                    else:
                        self._data[k_s] = float(v)
            except (json.JSONDecodeError, OSError):
                pass

    def _flush(self) -> None:
        out: dict[str, float] = {}
        for k, v in self._data.items():
            if isinstance(k, tuple) and len(k) == 2:
                tkr, sess = k
                sess_s = sess.isoformat() if hasattr(sess, "isoformat") else str(sess)
                out[f"{tkr}|{sess_s}"] = v
            else:
                out[str(k)] = v
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(out, f)
        os.replace(tmp, self._path)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = float(value)
        self._flush()

    def __delitem__(self, key):
        del self._data[key]
        self._flush()

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


__all__ = [
    "DEFAULT_FEATURE_KEY",
    "DEFAULT_MODEL_ROOT",
    "InferenceModel",
    "load_model",
    "open_mall",
    "predict_scores",
    "run_inference",
    "save_model",
    "train_and_save",
    "write_predictions_to_mall",
]
