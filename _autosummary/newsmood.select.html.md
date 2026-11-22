# newsmood.select

Feature selection + regularization tuning for the news panel.

These helpers are small and orthogonal — they can be composed in any order
before handing the result to [`newsmood.models.evaluate_walk_forward()`](newsmood.models.html.md#newsmood.models.evaluate_walk_forward).

- [`drop_low_variance()`](#newsmood.select.drop_low_variance): remove features whose stdev is below a cutoff
  (typically near-constant `q_*` columns from a thin news category).
- [`drop_high_correlation()`](#newsmood.select.drop_high_correlation): when two features have `|corr| > thresh`,
  drop the one with the lower correlation to the target — reduces collinearity.
- [`top_k_by_mutual_information()`](#newsmood.select.top_k_by_mutual_information): keep the K features with highest MI
  against the target. Robust to non-linear relationships.
- [`sweep_alpha()`](#newsmood.select.sweep_alpha): walk-forward Ridge sweep over a grid of `alpha` values,
  returning a DataFrame of per-fold and mean metrics for each alpha.

### Examples

```pycon
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
```

### Functions

| [`drop_low_variance`](#newsmood.select.drop_low_variance)(panel, \*[, target, ...])     | Drop features whose stdev is below `min_std`.                                                                    |
|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`drop_high_correlation`](#newsmood.select.drop_high_correlation)(panel, \*, target[, ...]) | Drop one of each pair of features with `|corr| > threshold`.                                                     |
| [`make_per_fold_mi_selector`](#newsmood.select.make_per_fold_mi_selector)([k, random_state])    | Return a closure `(train_df, target) -> [columns]` usable as `evaluate_walk_forward(..., feature_selector=...)`. |
| [`top_k_by_mutual_information`](#newsmood.select.top_k_by_mutual_information)(panel, \*, target)  | Keep the `k` features with highest mutual information against the target.                                        |
| [`sweep_alpha`](#newsmood.select.sweep_alpha)(panel, \*, target[, alphas, ...])   | Walk-forward Ridge sweep over a grid of `alpha` values.                                                          |

### newsmood.select.drop_high_correlation(panel, , target, threshold=0.95, feature_cols=None)

Drop one of each pair of features with `|corr| > threshold`.

The dropped column is the one with the **smaller** absolute Pearson
correlation to the target, so we preserve the more target-relevant of
each redundant pair.

* **Return type:**
  `DataFrame`

### newsmood.select.drop_low_variance(panel, , target=None, min_std=1e-06, feature_cols=None)

Drop features whose stdev is below `min_std`.

The target column is preserved. If `feature_cols` is None, every column
other than `target` is treated as a feature.

* **Return type:**
  `DataFrame`

### newsmood.select.make_per_fold_mi_selector(k=20, , random_state=0)

Return a closure `(train_df, target) -> [columns]` usable as
`evaluate_walk_forward(..., feature_selector=...)`.

Per-fold MI selection — runs only on the **train slice**, so OOS folds
can’t leak through feature choice. This is the leakage-free counterpart
to [`top_k_by_mutual_information()`](#newsmood.select.top_k_by_mutual_information) (which is global).

### newsmood.select.sweep_alpha(panel, , target, alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), feature_cols=None, n_splits=5, min_train=30, embargo=5)

Walk-forward Ridge sweep over a grid of `alpha` values.

Returns a DataFrame indexed by alpha with mean and std of IC, rank-IC,
sign accuracy, and Sharpe across folds.

* **Return type:**
  `DataFrame`

### newsmood.select.top_k_by_mutual_information(panel, , target, k=20, feature_cols=None, random_state=0)

Keep the `k` features with highest mutual information against the target.

Uses `sklearn.feature_selection.mutual_info_regression`. Robust to
monotonic non-linearities; ties broken by original column order.

* **Return type:**
  `DataFrame`
