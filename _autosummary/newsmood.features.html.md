# newsmood.features

Per-session features for news-driven prediction.

Given a news DataFrame with embeddings and a session anchor (from
[`newsmood.targets.attach_targets()`](newsmood.targets.html.md#newsmood.targets.attach_targets)), this module produces a session-level
feature frame ready for a regression / classification model:

- `news_count`: number of articles anchored to the session.
- `mean_vec_<i>`: `i`-th component of the mean embedding of articles
  anchored to the session.
- `cos_<seed>`: cosine similarity of the session’s mean embedding to a
  pre-defined seed prompt (interpretable scalar features).
- `q_<query>`: count of articles per query category (e.g.,
  `Earnings_Miss`, `Profit_Warning`).

The seed prompts themselves are embedded with the **same** embedder/cache as
the news, so the cosines are comparable. Seed embeddings hit the same cache
and so cost essentially nothing on re-run.

### Examples

```pycon
>>> from newsmood.features import DEFAULT_SEED_PROMPTS
>>> sorted(DEFAULT_SEED_PROMPTS)[:3]
['acquisition', 'bankruptcy', 'downgrade']
```

### Functions

| [`cosine_to_seeds`](#newsmood.features.cosine_to_seeds)(mean_vecs, seed_vecs)             | Cosine similarity between each row of `mean_vecs` and each seed.          |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`build_seed_embeddings`](#newsmood.features.build_seed_embeddings)(seeds, \*, cache[, ...])    | Embed each seed prompt (via the shared cache) — returns `{name: vector}`. |
| [`session_features`](#newsmood.features.session_features)(news_df, \*[, seed_vecs, ...])   | Group `news_df` by session, return one row per session.                   |
| [`panel_with_targets`](#newsmood.features.panel_with_targets)(news_df, \*[, seed_vecs, ...]) | Join per-session features with per-session targets.                       |

### newsmood.features.build_seed_embeddings(seeds, , cache, model=None, embedder=None)

Embed each seed prompt (via the shared cache) — returns `{name: vector}`.

Uses [`newsmood.embed.embed_doc_ids()`](newsmood.embed.html.md#newsmood.embed.embed_doc_ids) so the cache is shared with the
news embeddings.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]

### newsmood.features.cosine_to_seeds(mean_vecs, seed_vecs)

Cosine similarity between each row of `mean_vecs` and each seed.

Shapes: `mean_vecs (n, d)`, `seed_vecs (k, d)` → `(n, k)`.

* **Return type:**
  `ndarray`

### newsmood.features.panel_with_targets(news_df, , seed_vecs=None, target_cols=None, session_col='session', vec_col='vector', query_col='query', include_mean_vector_components=False, top_queries=None)

Join per-session features with per-session targets.

Targets are computed as the **first** observation of each target column
per session (returns at a session are session-level, so any row in the
group has the same value).

* **Return type:**
  `DataFrame`

### newsmood.features.session_features(news_df, , seed_vecs=None, session_col='session', vec_col='vector', query_col='query', include_mean_vector_components=False, top_queries=None)

Group `news_df` by session, return one row per session.

* **Parameters:**
  * **news_df** (`DataFrame`) – Must include `session_col`, `vec_col` (list-like vectors), and
    optionally `query_col`.
  * **seed_vecs** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`float`](https://docs.python.org/3/builtins/functions.html#float)]]]) – Mapping `{seed_name: vector}`. If given, a `cos_{seed_name}`
    column is added per seed (cosine of the session’s mean embedding to
    the seed).
  * **include_mean_vector_components** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – If True, also include `mean_vec_0 ... mean_vec_{d-1}` columns
    (mean embedding broadcast as scalar features). Off by default — the
    cosines are usually more interpretable and parsimonious.
  * **top_queries** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – If given, only count articles in this set of queries (else: counts
    every query observed).
* **Return type:**
  `DataFrame`
* **Returns:**
  DataFrame indexed by session (dates), one row per session.
