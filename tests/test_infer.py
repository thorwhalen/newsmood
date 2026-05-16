"""Tests for newsmood.infer — train/save/load/predict/write-to-mall."""

import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from newsmood.infer import (
    DEFAULT_FEATURE_KEY,
    InferenceModel,
    load_model,
    open_mall,
    predict_scores,
    run_inference,
    save_model,
    train_and_save,
    write_predictions_to_mall,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _make_panel(n_sessions: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    sessions = pd.Index(
        pd.date_range("2025-01-01", periods=n_sessions, freq="B").date,
        name="session",
    )
    f1 = rng.normal(size=n_sessions)
    f2 = rng.normal(size=n_sessions)
    target = 0.5 * f1 - 0.3 * f2 + rng.normal(scale=0.5, size=n_sessions)
    return pd.DataFrame(
        {
            "cos_a": f1,
            "cos_b": f2,
            "news_count": rng.integers(1, 5, size=n_sessions),
            "target": target,
        },
        index=sessions,
    )


# ---------------------------------------------------------------------------
# train_and_save / load_model
# ---------------------------------------------------------------------------


def test_train_and_save_roundtrip(tmp_path: Path):
    panel = _make_panel()
    model_path = tmp_path / "ridge.pkl"
    model = train_and_save(panel, target="target", model_path=str(model_path))
    assert model.target == "target"
    assert set(model.feature_cols) == {"cos_a", "cos_b", "news_count"}
    assert model.metadata["n_train"] > 0
    # Load roundtrip
    loaded = load_model(str(model_path))
    assert loaded.feature_cols == model.feature_cols
    assert loaded.target == "target"
    # Predictions match
    np.testing.assert_allclose(
        model.predict(panel), loaded.predict(panel), rtol=1e-12
    )


def test_train_no_features_raises():
    panel = pd.DataFrame({"target": [1.0, 2.0], "other": ["x", "y"]})
    with pytest.raises(ValueError, match="No feature columns"):
        train_and_save(panel, target="target")


def test_train_no_valid_rows_raises():
    panel = pd.DataFrame(
        {
            "cos_a": [None, None],
            "cos_b": [1.0, 2.0],
            "target": [None, None],
        }
    )
    with pytest.raises(ValueError, match="No rows"):
        train_and_save(panel, target="target")


# ---------------------------------------------------------------------------
# predict missing columns
# ---------------------------------------------------------------------------


def test_predict_missing_columns_raises():
    panel = _make_panel()
    model = train_and_save(panel, target="target")
    bad = panel.drop(columns=["cos_a"])
    with pytest.raises(ValueError, match="Feature columns missing"):
        model.predict(bad)


def test_predict_handles_nans_with_median_impute():
    panel = _make_panel(n_sessions=50)
    model = train_and_save(panel, target="target")
    test = panel.copy()
    test.loc[test.index[:5], "cos_a"] = np.nan
    preds = model.predict(test)
    # All predictions finite (NaN -> median-imputed)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# predict_scores
# ---------------------------------------------------------------------------


def test_predict_scores_returns_indexed_series():
    panel = _make_panel(n_sessions=30)
    model = train_and_save(panel, target="target")
    out = predict_scores(model, panel)
    assert isinstance(out, pd.Series)
    assert (out.index == panel.index).all()


# ---------------------------------------------------------------------------
# write_predictions_to_mall
# ---------------------------------------------------------------------------


def test_write_single_index_requires_ticker():
    s = pd.Series([0.1, 0.2], index=[date(2025, 3, 10), date(2025, 3, 11)])
    with pytest.raises(ValueError, match="pass ticker"):
        write_predictions_to_mall(s, mall={})


def test_write_single_index_with_ticker():
    s = pd.Series(
        [0.1, 0.2], index=[date(2025, 3, 10), date(2025, 3, 11)], name="score"
    )
    mall: dict = {}
    n = write_predictions_to_mall(s, mall=mall, ticker="SPY")
    assert n == 2
    feat = mall[DEFAULT_FEATURE_KEY]
    assert feat[("SPY", date(2025, 3, 10))] == pytest.approx(0.1)
    assert feat[("SPY", date(2025, 3, 11))] == pytest.approx(0.2)


def test_write_multiindex():
    idx = pd.MultiIndex.from_tuples(
        [("SPY", date(2025, 3, 10)), ("AAPL", date(2025, 3, 10))],
        names=["ticker", "session"],
    )
    s = pd.Series([0.1, -0.2], index=idx, name="score")
    mall: dict = {}
    n = write_predictions_to_mall(s, mall=mall)
    assert n == 2
    feat = mall[DEFAULT_FEATURE_KEY]
    assert feat[("SPY", date(2025, 3, 10))] == pytest.approx(0.1)
    assert feat[("AAPL", date(2025, 3, 10))] == pytest.approx(-0.2)


def test_write_skips_nan_values():
    s = pd.Series(
        [0.1, float("nan"), 0.3],
        index=[date(2025, 3, 10), date(2025, 3, 11), date(2025, 3, 12)],
    )
    mall: dict = {}
    n = write_predictions_to_mall(s, mall=mall, ticker="SPY")
    assert n == 2  # NaN skipped


def test_write_updates_existing_feature_store():
    existing = {("SPY", date(2025, 3, 10)): 99.0}
    mall = {DEFAULT_FEATURE_KEY: existing}
    s = pd.Series([0.5], index=[date(2025, 3, 11)])
    write_predictions_to_mall(s, mall=mall, ticker="SPY")
    # Old entry preserved, new entry added
    assert mall[DEFAULT_FEATURE_KEY][("SPY", date(2025, 3, 10))] == 99.0
    assert mall[DEFAULT_FEATURE_KEY][("SPY", date(2025, 3, 11))] == 0.5


# ---------------------------------------------------------------------------
# run_inference end-to-end
# ---------------------------------------------------------------------------


def test_run_inference_writes_predictions(tmp_path: Path):
    panel = _make_panel(n_sessions=30)
    model = train_and_save(panel, target="target")
    mall: dict = {}
    n = run_inference(panel, model=model, mall=mall, ticker="SPY")
    assert n == 30
    feat = mall[DEFAULT_FEATURE_KEY]
    # Every session key present
    assert all(date(2025, 1, 1) <= sess for (_, sess) in feat.keys())


# ---------------------------------------------------------------------------
# open_mall persistence
# ---------------------------------------------------------------------------


def test_open_mall_roundtrip(tmp_path: Path):
    mall = open_mall(str(tmp_path / "mall"))
    sub = {}
    mall[DEFAULT_FEATURE_KEY] = sub
    mall[DEFAULT_FEATURE_KEY][("SPY", date(2025, 3, 10))] = 0.42
    mall[DEFAULT_FEATURE_KEY][("AAPL", date(2025, 3, 10))] = -0.01
    # Re-open
    mall2 = open_mall(str(tmp_path / "mall"))
    assert mall2[DEFAULT_FEATURE_KEY][("SPY", date(2025, 3, 10))] == pytest.approx(0.42)
    assert mall2[DEFAULT_FEATURE_KEY][("AAPL", date(2025, 3, 10))] == pytest.approx(-0.01)


def test_open_mall_missing_key():
    mall = open_mall()
    with pytest.raises(KeyError):
        mall["nonexistent_key_xyz_12345"]


# ---------------------------------------------------------------------------
# Strategy integration: smoke test
# ---------------------------------------------------------------------------


def test_inference_writes_keys_strategy_can_read():
    """The score we write must use keys the hedger news_embed strategy can find."""
    from datetime import datetime, timezone

    panel = _make_panel(n_sessions=10)
    model = train_and_save(panel, target="target")
    mall: dict = {}
    run_inference(panel, model=model, mall=mall, ticker="SPY")

    # Simulate what hedger.strategies.news_embed.news_embed does
    test_session = panel.index[-1]
    feat = mall[DEFAULT_FEATURE_KEY]
    key = ("SPY", test_session)
    assert key in feat
    assert isinstance(feat[key], float)
