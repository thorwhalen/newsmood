"""Per-session features for news-driven prediction.

Given a news DataFrame with embeddings and a session anchor (from
:func:`newsmood.targets.attach_targets`), this module produces a session-level
feature frame ready for a regression / classification model:

- ``news_count``: number of articles anchored to the session.
- ``mean_vec_<i>``: ``i``-th component of the mean embedding of articles
  anchored to the session.
- ``cos_<seed>``: cosine similarity of the session's mean embedding to a
  pre-defined seed prompt (interpretable scalar features).
- ``q_<query>``: count of articles per query category (e.g.,
  ``Earnings_Miss``, ``Profit_Warning``).

The seed prompts themselves are embedded with the **same** embedder/cache as
the news, so the cosines are comparable. Seed embeddings hit the same cache
and so cost essentially nothing on re-run.

Examples
--------
>>> from newsmood.features import DEFAULT_SEED_PROMPTS
>>> sorted(DEFAULT_SEED_PROMPTS)[:3]
['acquisition', 'bankruptcy', 'downgrade']
"""

from collections.abc import Mapping
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from newsmood.ingest import make_doc_id


DEFAULT_SEED_PROMPTS: dict[str, str] = {
    "earnings_miss": "Company reports an earnings miss; profit and guidance fall short of expectations.",
    "profit_warning": "Company issues a profit warning ahead of its earnings release.",
    "acquisition": "An acquisition or merger is announced between two companies.",
    "bankruptcy": "A major company files for bankruptcy or defaults on its debt.",
    "leadership_change": "A CEO resignation or significant change in executive leadership.",
    "downgrade": "An analyst downgrades a stock or lowers its price target.",
    "upgrade": "An analyst upgrades a stock or raises its price target.",
    "tariffs": "New tariffs, trade restrictions, or escalation of a trade war.",
    "regulation": "New regulations or government investigations affecting industry.",
    "rate_hike": "The central bank hikes interest rates or signals a hawkish stance.",
    "rate_cut": "The central bank cuts interest rates or signals a dovish stance.",
    "inflation_high": "Inflation comes in hotter than expected; CPI surprises to the upside.",
    "recession": "Recession fears or weakening macroeconomic conditions.",
    "geopolitical": "Geopolitical crisis, war, or major political instability.",
}


# -- Embedding-vector helpers (numpy-backed) ---------------------------------


def _as_matrix(vectors: pd.Series) -> Optional[np.ndarray]:
    """Stack a Series of vectors into an ``(n, d)`` matrix; ``None`` if all empty."""
    arrs = [np.asarray(v, dtype=float) for v in vectors if v is not None]
    if not arrs:
        return None
    dim = len(arrs[0])
    if any(len(a) != dim for a in arrs):
        raise ValueError("Mixed embedding dimensions in input")
    return np.vstack(arrs)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return x / n


def cosine_to_seeds(
    mean_vecs: np.ndarray, seed_vecs: np.ndarray
) -> np.ndarray:
    """Cosine similarity between each row of ``mean_vecs`` and each seed.

    Shapes: ``mean_vecs (n, d)``, ``seed_vecs (k, d)`` → ``(n, k)``.
    """
    return _l2_normalize(mean_vecs) @ _l2_normalize(seed_vecs).T


# -- Seed-embedding helper ---------------------------------------------------


def build_seed_embeddings(
    seeds: Mapping[str, str],
    *,
    cache,
    model: Optional[str] = None,
    embedder=None,
) -> dict[str, list[float]]:
    """Embed each seed prompt (via the shared cache) — returns ``{name: vector}``.

    Uses :func:`newsmood.embed.embed_doc_ids` so the cache is shared with the
    news embeddings.
    """
    from newsmood.embed import embed_doc_ids

    pairs = [(make_doc_id(text), text) for text in seeds.values()]
    id_to_vec = embed_doc_ids(pairs, cache=cache, model=model, embedder=embedder)
    return {name: id_to_vec[make_doc_id(text)] for name, text in seeds.items()}


# -- Session-level aggregation ----------------------------------------------


def session_features(
    news_df: pd.DataFrame,
    *,
    seed_vecs: Optional[Mapping[str, list[float]]] = None,
    session_col: str = "session",
    vec_col: str = "vector",
    query_col: str = "query",
    include_mean_vector_components: bool = False,
    top_queries: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Group ``news_df`` by session, return one row per session.

    Parameters
    ----------
    news_df
        Must include ``session_col``, ``vec_col`` (list-like vectors), and
        optionally ``query_col``.
    seed_vecs
        Mapping ``{seed_name: vector}``. If given, a ``cos_{seed_name}``
        column is added per seed (cosine of the session's mean embedding to
        the seed).
    include_mean_vector_components
        If True, also include ``mean_vec_0 ... mean_vec_{d-1}`` columns
        (mean embedding broadcast as scalar features). Off by default — the
        cosines are usually more interpretable and parsimonious.
    top_queries
        If given, only count articles in this set of queries (else: counts
        every query observed).

    Returns
    -------
    DataFrame indexed by session (dates), one row per session.
    """
    if news_df.empty:
        return pd.DataFrame()

    valid = news_df.dropna(subset=[session_col, vec_col])
    if valid.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    seed_names: list[str] = []
    seed_matrix: Optional[np.ndarray] = None
    if seed_vecs:
        seed_names = list(seed_vecs.keys())
        seed_matrix = np.vstack([np.asarray(v, dtype=float) for v in seed_vecs.values()])

    queries_in_data: list[str] = []
    if query_col in valid.columns:
        if top_queries is not None:
            queries_in_data = sorted(top_queries)
        else:
            queries_in_data = sorted(
                q for q in valid[query_col].dropna().unique() if q
            )

    for session, group in valid.groupby(session_col):
        mat = _as_matrix(group[vec_col])
        if mat is None:
            continue
        mean_vec = mat.mean(axis=0)
        row: dict = {
            "session": session,
            "news_count": int(len(group)),
        }
        if seed_matrix is not None:
            cos = cosine_to_seeds(mean_vec[None, :], seed_matrix).ravel()
            for name, val in zip(seed_names, cos):
                row[f"cos_{name}"] = float(val)
        if include_mean_vector_components:
            for i, v in enumerate(mean_vec):
                row[f"mean_vec_{i}"] = float(v)
        if queries_in_data:
            counts = group[query_col].value_counts()
            for q in queries_in_data:
                row[f"q_{q}"] = int(counts.get(q, 0))
        out_rows.append(row)

    return pd.DataFrame(out_rows).set_index("session").sort_index()


def panel_with_targets(
    news_df: pd.DataFrame,
    *,
    seed_vecs: Optional[Mapping[str, list[float]]] = None,
    target_cols: Optional[Iterable[str]] = None,
    session_col: str = "session",
    vec_col: str = "vector",
    query_col: str = "query",
    include_mean_vector_components: bool = False,
    top_queries: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Join per-session features with per-session targets.

    Targets are computed as the **first** observation of each target column
    per session (returns at a session are session-level, so any row in the
    group has the same value).
    """
    features = session_features(
        news_df,
        seed_vecs=seed_vecs,
        session_col=session_col,
        vec_col=vec_col,
        query_col=query_col,
        include_mean_vector_components=include_mean_vector_components,
        top_queries=top_queries,
    )
    if features.empty:
        return features

    if target_cols is None:
        target_cols = [
            c
            for c in news_df.columns
            if c.endswith("d") and ("_ret_" in c or c.startswith("ret_"))
        ]
    target_cols = list(target_cols)
    if not target_cols:
        return features

    targets = (
        news_df.dropna(subset=[session_col])
        .groupby(session_col)[target_cols]
        .first()
    )
    return features.join(targets, how="left")


__all__ = [
    "DEFAULT_SEED_PROMPTS",
    "cosine_to_seeds",
    "build_seed_embeddings",
    "session_features",
    "panel_with_targets",
]
