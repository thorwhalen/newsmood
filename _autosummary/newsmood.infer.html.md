# newsmood.infer

Train, persist, and apply a news-embedding signal model.

Two halves:

- [`train_and_save()`](#newsmood.infer.train_and_save): fit the baseline Ridge pipeline on a labeled panel,
  pickle the fitted pipeline plus the feature schema and a model version.
- [`load_model()`](#newsmood.infer.load_model) / [`predict_scores()`](#newsmood.infer.predict_scores) / [`write_predictions_to_mall()`](#newsmood.infer.write_predictions_to_mall):
  consume the persisted model to score new feature frames, then write per-
  `(symbol, session)` scores into a mall the
  `hedger.strategies.news_embed` strategy reads from.

The mall is any `MutableMapping` — a dict for tests, a `dol` store for
production. The persisted feature_key is `features:news_embed_v1` by default
(matches the strategy’s default).

### Functions

| [`load_model`](#newsmood.infer.load_model)(model_path)                            | Load an InferenceModel produced by [`save_model()`](#newsmood.infer.save_model).   |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| [`open_mall`](#newsmood.infer.open_mall)([root])                                 | Open a persistent mall — a dol-backed JSON store at `root`.                                         |
| [`predict_scores`](#newsmood.infer.predict_scores)(model, feature_df)                 | Apply model to feature_df; return Series indexed like feature_df.                                   |
| [`run_inference`](#newsmood.infer.run_inference)(feature_df, \*, model, mall[, ...]) | End-to-end: predict on `feature_df` and write to `mall`.                                            |
| [`save_model`](#newsmood.infer.save_model)(model, model_path)                     | Persist an InferenceModel to disk (pickle).                                                         |
| [`train_and_save`](#newsmood.infer.train_and_save)(panel, \*, target[, ...])          | Fit on the full panel (no CV; CV is for evaluation) and persist.                                    |
| [`write_predictions_to_mall`](#newsmood.infer.write_predictions_to_mall)(predictions, \*, mall)  | Write per-(ticker, session) scores into `mall[feature_key]`.                                        |

### Classes

| [`InferenceModel`](#newsmood.infer.InferenceModel)(pipeline, feature_cols, target)   | A fitted Ridge pipeline plus the schema needed to apply it.   |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------|

### *class* newsmood.infer.InferenceModel(pipeline, feature_cols, target, version='v1', metadata=dict())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A fitted Ridge pipeline plus the schema needed to apply it.

#### pipeline

A fitted sklearn `Pipeline` (typically `StandardScaler → Ridge`).

* **Type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

#### feature_cols

Column names — both required and order-sensitive.

* **Type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

#### target

Target column the model was trained against (for provenance).

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### version

A free-form version string (e.g. `"v1"`, `"2026-05-15"`).

* **Type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

#### metadata

Anything else worth keeping: training row count, IC summary, etc.

* **Type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

#### predict(X)

Predict on a feature frame; missing feature columns raise.

* **Return type:**
  `ndarray`

### newsmood.infer.load_model(model_path)

Load an InferenceModel produced by [`save_model()`](#newsmood.infer.save_model).

* **Return type:**
  [`InferenceModel`](#newsmood.infer.InferenceModel)

### newsmood.infer.open_mall(root=None)

Open a persistent mall — a dol-backed JSON store at `root`.

Each top-level key (e.g. `features:news_embed_v1`) maps to a JSON-blob
file at `root`. Nested entries are JSON-coded `[ticker, session-iso] -> float`
pairs.

For a quick in-process mall, just pass a `dict()` everywhere — this
helper is for daemon-style daily inference jobs that need to persist.

* **Return type:**
  [`MutableMapping`](https://docs.python.org/3/library/typing.html#typing.MutableMapping)

### newsmood.infer.predict_scores(model, feature_df)

Apply model to feature_df; return Series indexed like feature_df.

* **Return type:**
  `Series`

### newsmood.infer.run_inference(feature_df, , model, mall, ticker=None, feature_key='features:news_embed_v1')

End-to-end: predict on `feature_df` and write to `mall`.

Returns the number of mall entries written.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### newsmood.infer.save_model(model, model_path)

Persist an InferenceModel to disk (pickle).

* **Return type:**
  [`None`](https://docs.python.org/3/builtins/constants.html#None)

### newsmood.infer.train_and_save(panel, , target, feature_cols=None, alpha=1.0, model_path=None, version='v1', metadata=None)

Fit on the full panel (no CV; CV is for evaluation) and persist.

The CV pipeline lives in [`newsmood.models.evaluate_walk_forward()`](newsmood.models.html.md#newsmood.models.evaluate_walk_forward) —
that’s where you decide *whether* the model has signal. Once you have
chosen `alpha` / feature set there, run *this* function on the same
panel to produce the operational artifact.

* **Return type:**
  [`InferenceModel`](#newsmood.infer.InferenceModel)

### newsmood.infer.write_predictions_to_mall(predictions, , mall, ticker=None, feature_key='features:news_embed_v1')

Write per-(ticker, session) scores into `mall[feature_key]`.

Two index shapes are supported:

- Single Index of session dates: `ticker` must be provided; every entry
  is keyed `(ticker, session)`.
- MultiIndex of `(ticker, session)`: `ticker` is ignored.

`mall[feature_key]` is created (as a plain `dict`) if absent, or
updated in place if present.

Returns the number of entries written.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)
