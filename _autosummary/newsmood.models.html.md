# newsmood.models

Baseline walk-forward modeling on the news-feature panel.

A deliberately small surface aimed at the first vertical slice:

- [`walk_forward_splits()`](#newsmood.models.walk_forward_splits): expanding-window splits with an embargo, so
  no test fold can leak information from training (forward-return overlap).
- [`information_coefficient()`](#newsmood.models.information_coefficient) / [`rank_information_coefficient()`](#newsmood.models.rank_information_coefficient):
  the two metrics the user’s hedger/RESEARCH.md cites as the canonical
  factor-research scoring methods.
- [`fit_predict_ridge()`](#newsmood.models.fit_predict_ridge): single fold fit-and-predict.
- [`evaluate_walk_forward()`](#newsmood.models.evaluate_walk_forward): glues the two together; returns per-fold
  metrics + a long-form DataFrame of out-of-sample predictions.

Out of scope here (deferred): hyperparameter tuning, classification head,
panel models with ticker fixed effects, RL bolt-on. The current code is
intentionally a baseline — it answers “is there any usable signal?”.

### Functions

| [`evaluate_walk_forward`](#newsmood.models.evaluate_walk_forward)(panel, \*, target[, ...])   | Walk-forward Ridge baseline over a session-level panel.      |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [`fit_predict_ridge`](#newsmood.models.fit_predict_ridge)(X_train, y_train, X_test, \*)   | Standard-scale + Ridge fit, return test predictions.         |
| [`information_coefficient`](#newsmood.models.information_coefficient)(y_true, y_pred)           | Pearson IC.                                                  |
| [`long_short_sharpe`](#newsmood.models.long_short_sharpe)(y_true, y_pred, \*[, cost_bps]) | Sharpe of a sign-based long-short portfolio.                 |
| [`rank_information_coefficient`](#newsmood.models.rank_information_coefficient)(y_true, y_pred)      | Spearman IC.                                                 |
| [`sign_accuracy`](#newsmood.models.sign_accuracy)(y_true, y_pred)                     | Fraction of predictions whose sign matches the true sign.    |
| [`walk_forward_splits`](#newsmood.models.walk_forward_splits)(n, \*[, n_splits, ...])       | Yield `(train_idx, test_idx)` pairs for expanding-window CV. |

### Classes

| [`WalkForwardResult`](#newsmood.models.WalkForwardResult)(target, per_fold, predictions)   | Output container for [`evaluate_walk_forward()`](#newsmood.models.evaluate_walk_forward).   |
|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|

### *class* newsmood.models.WalkForwardResult(target, per_fold, predictions, feature_names=list())

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Output container for [`evaluate_walk_forward()`](#newsmood.models.evaluate_walk_forward).

#### summary()

Aggregate metrics across folds (mean).

* **Return type:**
  `Series`

### newsmood.models.evaluate_walk_forward(panel, , target, feature_cols=None, n_splits=5, min_train=30, embargo=1, alpha=1.0, cost_bps=0.0, feature_selector=None)

Walk-forward Ridge baseline over a session-level panel.

The panel must be sorted by session ascending. Rows with NaN target or
feature values are dropped before training.

* **Parameters:**
  **feature_selector** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable)[[`DataFrame`, [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]]) – Optional callable `(train_df, target_name) -> list[selected_features]`.
  Called *inside each fold* on the **train slice only** — this avoids the
  look-ahead leakage that occurs when feature selection is run globally
  before splitting. The selector chooses among `feature_cols`.
* **Return type:**
  [`WalkForwardResult`](#newsmood.models.WalkForwardResult)

### newsmood.models.fit_predict_ridge(X_train, y_train, X_test, , alpha=1.0)

Standard-scale + Ridge fit, return test predictions.

* **Return type:**
  `ndarray`

### newsmood.models.information_coefficient(y_true, y_pred)

Pearson IC. `NaN`-tolerant; returns `nan` if insufficient data.

```pycon
>>> round(information_coefficient([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 6)
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.models.long_short_sharpe(y_true, y_pred, , cost_bps=0.0)

Sharpe of a sign-based long-short portfolio.

For each session, go long if `y_pred > 0`, short if `y_pred < 0`, scale
by 1; realized return is `sign(y_pred) * y_true`. Optional per-trade
transaction cost in basis points is subtracted when the sign flips.

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.models.rank_information_coefficient(y_true, y_pred)

Spearman IC. `NaN`-tolerant.

```pycon
>>> round(rank_information_coefficient([1, 2, 3], [3, 2, 1]), 6)
-1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.models.sign_accuracy(y_true, y_pred)

Fraction of predictions whose sign matches the true sign. Zero-true rows skipped.

```pycon
>>> sign_accuracy([1, -1, 1], [0.5, -0.2, 0.3])
1.0
```

* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)

### newsmood.models.walk_forward_splits(n, , n_splits=5, min_train=30, embargo=1, test_size=None)

Yield `(train_idx, test_idx)` pairs for expanding-window CV.

Each test fold begins `embargo` indices *after* the last train index, so
the forward-return horizon — which makes label `r_t` peek at price
`t + h` — cannot leak across the train/test boundary as long as
`embargo >= h`.

* **Parameters:**
  * **n** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Total number of (already chronologically-ordered) rows.
  * **n_splits** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – How many folds to produce.
  * **min_train** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Minimum number of rows in the first training fold.
  * **embargo** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Gap (in rows / sessions) between train and test.
  * **test_size** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/builtins/functions.html#int)]) – Size of each test fold. Default: `(n - min_train) // n_splits`.
  * **list****(****walk_forward_splits****(****20** ( *>>> splits =*)
  * **n_splits=3**
  * **min_train=8**
  * **embargo=1**
  * **test_size=3****)****)**
  * **len****(****splits****)** ( *>>>*)
  * **3**
  * **splits****[****0****]****[****1****]****.tolist****(****)** ( *>>>*)
  * **[****9**
  * **10**
  * **11****]**
* **Return type:**
  [`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[`ndarray`, `ndarray`]]
