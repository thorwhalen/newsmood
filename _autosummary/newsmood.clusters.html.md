# newsmood.clusters

Cluster-bucket features.

Fits a K-Means codebook on a calibration window of embeddings, then exposes
that codebook for two things:

1. Labeling each article with a `cluster` ID (deterministic).
2. Producing per-session cluster-count features (one column per cluster).

The codebook itself is fit via `imbed.components.clusterization.fit_kmeans()`
(an enhancement we added to `imbed` specifically for this — it returns a
fitted model that can predict on new vectors, not just a one-shot labeling
of the training set).

### Examples

```pycon
>>> from newsmood.clusters import fit_news_clusters, attach_cluster_id
>>> from newsmood.embed import deterministic_dummy_embedder
>>> import pandas as pd, numpy as np
>>> e = deterministic_dummy_embedder(dim=8)
>>> df = pd.DataFrame({'doc_id': list('abcd'),
...                    'vector': [e('a'), e('b'), e('c'), e('d')]})
>>> model = fit_news_clusters(df, n_clusters=2, random_state=0)
>>> labeled = attach_cluster_id(df, model)
>>> sorted(labeled['cluster'].unique().tolist())
[0, 1]
```

### Functions

| [`fit_news_clusters`](#newsmood.clusters.fit_news_clusters)(df, \*[, n_clusters, ...])      | Fit a K-Means codebook on `df[vec_col]`.                                          |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`attach_cluster_id`](#newsmood.clusters.attach_cluster_id)(df, model, \*[, vec_col, ...])  | Return a copy of `df` with cluster IDs in `out_col`.                              |
| [`session_cluster_counts`](#newsmood.clusters.session_cluster_counts)(df, \*, n_clusters[, ...]) | Per-session counts of articles in each cluster (zero-padded across all clusters). |

### newsmood.clusters.attach_cluster_id(df, model, , vec_col='vector', out_col='cluster')

Return a copy of `df` with cluster IDs in `out_col`.

Rows whose vector is missing get `-1`.

* **Return type:**
  `DataFrame`

### newsmood.clusters.fit_news_clusters(df, , n_clusters=64, vec_col='vector', sample_size=None, random_state=None, normalize=True, \*\*kmeans_kwargs)

Fit a K-Means codebook on `df[vec_col]`.

* **Parameters:**
  * **df** (`DataFrame`) – Must have `vec_col` of list-like vectors (rows with `None` are
    skipped).
  * **n_clusters** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Codebook size. 32-128 is typical; 64 is a reasonable default for
    ~hundreds-of-thousands of embeddings.
  * **sample_size** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – If set, randomly sample this many rows for fitting (full corpus is
    rarely needed; sampling speeds up KMeans dramatically).
  * **random_state** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Reproducibility seed.
  * **normalize** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – L2-normalize before fitting so Euclidean distance approximates cosine
    — appropriate for semantic embeddings.
  * **\*\*kmeans_kwargs** – Forwarded to `sklearn.cluster.KMeans`.
* **Returns:**
  Fitted model (callable on new vectors).

### newsmood.clusters.session_cluster_counts(df, , n_clusters, session_col='session', cluster_col='cluster', column_prefix='clust_')

Per-session counts of articles in each cluster (zero-padded across all clusters).

Returns a DataFrame indexed by session with `n_clusters` columns:
`{column_prefix}0` … `{column_prefix}{n_clusters-1}`.

* **Return type:**
  `DataFrame`
