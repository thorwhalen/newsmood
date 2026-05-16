"""Comprehensive evaluation with the new tooling.

What this runs:
1. Ingest + load embeddings from cache (only embedded rows kept).
2. Cluster (k=32) + attach SPY targets + build session-level feature panel.
3. SPY alpha sweep on the full feature set.
4. SPY alpha sweep with **per-fold MI selection** (leakage-free).
5. SPY backtest with best alpha + tanh sizing + 1bp costs.
6. Per-ticker panel + per-ticker walk-forward + backtest, ranked by IC.

Reports IC, rank-IC, sign accuracy, Sharpe, max drawdown, win rate, hit
rate, turnover.

Usage:
    python misc/full_evaluation.py [--since DATE] [--until DATE] [--horizon N]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings


def main(args) -> int:
    warnings.filterwarnings("ignore")
    import dol
    import pandas as pd

    from newsmood import (
        DEFAULT_SEED_PROMPTS,
        DEFAULT_SEARCHES_ROOT,
        attach_cluster_id,
        attach_targets,
        attach_targets_per_ticker,
        backtest_walk_forward,
        build_seed_embeddings,
        default_embedding_cache,
        evaluate_walk_forward,
        fit_news_clusters,
        ingest_searches,
        make_per_fold_mi_selector,
        panel_with_targets,
        per_ticker_evaluate,
        per_ticker_panel,
        session_cluster_counts,
        sweep_alpha,
    )

    cache = default_embedding_cache()
    print(f"cache: {len(list(cache)):,} embeddings", flush=True)

    t0 = time.time()
    store = dol.JsonFiles(DEFAULT_SEARCHES_ROOT)
    df = ingest_searches(store, source="newsdata", since=args.since, until=args.until)
    cached_set = set(cache)
    df = df[df["doc_id"].isin(cached_set)].copy()
    df["vector"] = df["doc_id"].map(lambda x: cache[x])
    print(f"usable rows: {len(df):,}  ({time.time()-t0:.1f}s)", flush=True)
    if df.empty:
        print("No embedded rows in window — run embed step first.")
        return 1

    t0 = time.time()
    model_c = fit_news_clusters(df, n_clusters=args.n_clusters, random_state=0,
                                 sample_size=min(50_000, len(df)))
    df = attach_cluster_id(df, model_c)
    df = attach_targets(df, tickers=("SPY",), horizons=(args.horizon,))
    seed_vecs = build_seed_embeddings(DEFAULT_SEED_PROMPTS, cache=cache)
    target = f"SPY_ret_{args.horizon}d"
    panel = panel_with_targets(df, seed_vecs=seed_vecs, target_cols=[target])
    cluster_counts = session_cluster_counts(df, n_clusters=args.n_clusters)
    panel = panel.join(cluster_counts, how="left").fillna(0)
    print(f"SPY panel: {panel.shape}  ({time.time()-t0:.1f}s)", flush=True)

    # 3. SPY alpha sweep (full features)
    print()
    print("=" * 78)
    print(f"SPY {target}: alpha sweep on full feature set ({panel.shape[1]-1} features)")
    print("=" * 78)
    sweep = sweep_alpha(panel, target=target,
                       alphas=(0.1, 1.0, 10.0, 100.0, 1000.0),
                       n_splits=5, min_train=max(30, len(panel) // 5), embargo=5)
    print(sweep.to_string(float_format=lambda v: f"{v: .4f}"))

    # 4. With per-fold MI selection (leakage-free)
    print()
    print("=" * 78)
    print(f"SPY {target}: per-fold MI top-20 (leakage-free)")
    print("=" * 78)
    selector = make_per_fold_mi_selector(k=20, random_state=0)
    best_alpha = None
    best_ic = -1e9
    rows = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        res = evaluate_walk_forward(
            panel, target=target, n_splits=5, min_train=max(30, len(panel) // 5),
            embargo=5, alpha=alpha, feature_selector=selector,
        )
        if res.per_fold.empty:
            continue
        ic = float(res.per_fold["ic"].mean())
        rows.append({
            "alpha": alpha,
            "mean_ic": ic,
            "mean_rank_ic": float(res.per_fold["rank_ic"].mean()),
            "mean_sign_acc": float(res.per_fold["sign_acc"].mean()),
            "mean_sharpe": float(res.per_fold["sharpe"].mean()),
        })
        if ic > best_ic:
            best_ic = ic
            best_alpha = alpha
            best_res = res
    print(pd.DataFrame(rows).set_index("alpha").to_string(float_format=lambda v: f"{v: .4f}"))

    # 5. SPY backtest at best alpha
    if best_alpha is not None:
        print()
        print("=" * 78)
        print(f"SPY OOS backtest at α={best_alpha} (per-fold MI, tanh sizing, 1bp costs)")
        print("=" * 78)
        bt = backtest_walk_forward(best_res.predictions, sizing="tanh", cost_bps=1.0)
        print(bt.summary().to_string(float_format=lambda v: f"{v: .4f}"))

    # 6. Per-ticker evaluation
    print()
    print("=" * 78)
    print(f"Per-ticker walk-forward (ret_{args.horizon}d)")
    print("=" * 78)
    t0 = time.time()
    pt_panel = per_ticker_panel(df, seed_vecs=seed_vecs, min_articles_per_ticker=args.min_per_ticker)
    if pt_panel.empty:
        print("  no tickers met min-articles threshold")
        return 0
    pt_panel = attach_targets_per_ticker(pt_panel, horizons=(args.horizon,))
    print(f"per-ticker panel: {pt_panel.shape}  ({time.time()-t0:.1f}s)", flush=True)
    if pt_panel.empty or f"ret_{args.horizon}d" not in pt_panel.columns:
        return 0

    pt_results = per_ticker_evaluate(
        pt_panel,
        target=f"ret_{args.horizon}d",
        n_splits=5,
        min_train=20,
        embargo=2,
        alpha=10.0,
        cost_bps=1.0,
    )
    if pt_results.empty:
        print("  no per-ticker results (panels too short)")
        return 0
    print(pt_results.to_string(float_format=lambda v: f"{v: .4f}"))

    # Aggregate
    print()
    print("--- aggregate across tickers ---")
    n_pos_ic = int((pt_results["mean_ic"] > 0).sum())
    print(f"tickers with positive mean_ic:    {n_pos_ic} / {len(pt_results)}")
    n_pos_sharpe = int((pt_results.get("bt_sharpe", pd.Series(dtype=float)) > 0).sum())
    print(f"tickers with positive bt_sharpe:  {n_pos_sharpe} / {len(pt_results)}")
    if "bt_final_equity" in pt_results.columns:
        equal_w = float(pt_results["bt_final_equity"].mean())
        print(f"equal-weight final equity (OOS):  {equal_w:.4f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2025-01-01")
    parser.add_argument("--until", default="2026-06-01")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--n-clusters", type=int, default=32)
    parser.add_argument("--min-per-ticker", type=int, default=100)
    args = parser.parse_args()
    sys.exit(main(args))
