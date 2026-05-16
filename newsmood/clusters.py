"""Cluster-bucket features.

Fits a K-Means codebook on a calibration window of embeddings, then exposes
that codebook for two things:

1. Labeling each article with a ``cluster`` ID (deterministic).
2. Producing per-session cluster-count features (one column per cluster).

The codebook itself is fit via :func:`imbed.components.clusterization.fit_kmeans`
(an enhancement we added to ``imbed`` specifically for this — it returns a
fitted model that can predict on new vectors, not just a one-shot labeling
of the training set).

Examples
--------
>>> from newsmood.clusters import fit_news_clusters, attach_cluster_id  # doctest: +SKIP
>>> from newsmood.embed import deterministic_dummy_embedder              # doctest: +SKIP
>>> import pandas as pd, numpy as np                                      # doctest: +SKIP
>>> e = deterministic_dummy_embedder(dim=8)                               # doctest: +SKIP
>>> df = pd.DataFrame({'doc_id': list('abcd'),
...                    'vector': [e('a'), e('b'), e('c'), e('d')]})       # doctest: +SKIP
>>> model = fit_news_clusters(df, n_clusters=2, random_state=0)           # doctest: +SKIP
>>> labeled = attach_cluster_id(df, model)                                # doctest: +SKIP
>>> sorted(labeled['cluster'].unique().tolist())                          # doctest: +SKIP
[0, 1]
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def fit_news_clusters(
    df: pd.DataFrame,
    *,
    n_clusters: int = 64,
    vec_col: str = "vector",
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
    normalize: bool = True,
    **kmeans_kwargs,
):
    """Fit a K-Means codebook on ``df[vec_col]``.

    Parameters
    ----------
    df
        Must have ``vec_col`` of list-like vectors (rows with ``None`` are
        skipped).
    n_clusters
        Codebook size. 32-128 is typical; 64 is a reasonable default for
        ~hundreds-of-thousands of embeddings.
    sample_size
        If set, randomly sample this many rows for fitting (full corpus is
        rarely needed; sampling speeds up KMeans dramatically).
    random_state
        Reproducibility seed.
    normalize
        L2-normalize before fitting so Euclidean distance approximates cosine
        — appropriate for semantic embeddings.
    **kmeans_kwargs
        Forwarded to ``sklearn.cluster.KMeans``.

    Returns
    -------
    Fitted model (callable on new vectors).
    """
    from imbed.components.clusterization import fit_kmeans

    valid = df.dropna(subset=[vec_col])
    if valid.empty:
        raise ValueError("No vectors to fit cluster model on")
    if sample_size is not None and len(valid) > sample_size:
        valid = valid.sample(n=sample_size, random_state=random_state)
    vectors = np.array(list(valid[vec_col].values))
    return fit_kmeans(
        vectors,
        n_clusters=n_clusters,
        normalize=normalize,
        random_state=random_state,
        **kmeans_kwargs,
    )


def attach_cluster_id(
    df: pd.DataFrame,
    model,
    *,
    vec_col: str = "vector",
    out_col: str = "cluster",
) -> pd.DataFrame:
    """Return a copy of ``df`` with cluster IDs in ``out_col``.

    Rows whose vector is missing get ``-1``.
    """
    out = df.copy()
    has_vec = out[vec_col].notna()
    if not has_vec.any():
        out[out_col] = -1
        return out
    vectors = np.array(list(out.loc[has_vec, vec_col].values))
    labels = model.predict(vectors)
    out[out_col] = -1
    out.loc[has_vec, out_col] = labels
    return out


def session_cluster_counts(
    df: pd.DataFrame,
    *,
    n_clusters: int,
    session_col: str = "session",
    cluster_col: str = "cluster",
    column_prefix: str = "clust_",
) -> pd.DataFrame:
    """Per-session counts of articles in each cluster (zero-padded across all clusters).

    Returns a DataFrame indexed by session with ``n_clusters`` columns:
    ``{column_prefix}0`` ... ``{column_prefix}{n_clusters-1}``.
    """
    if df.empty:
        return pd.DataFrame()
    valid = df.dropna(subset=[session_col, cluster_col])
    valid = valid[valid[cluster_col] >= 0]
    if valid.empty:
        return pd.DataFrame()
    counts = (
        valid.groupby([session_col, cluster_col])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all cluster columns exist (zero-pad)
    for i in range(n_clusters):
        if i not in counts.columns:
            counts[i] = 0
    counts = counts[sorted(counts.columns)]
    counts.columns = [f"{column_prefix}{int(c)}" for c in counts.columns]
    return counts.sort_index()


__all__ = [
    "fit_news_clusters",
    "attach_cluster_id",
    "session_cluster_counts",
]
