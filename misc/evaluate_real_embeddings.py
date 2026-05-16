"""Run the full pipeline with real (cached) embeddings and report IC.

Assumes embeddings already exist in ``default_embedding_cache()``. If not,
articles whose ``doc_id`` is missing from the cache are dropped.

Usage:
    python misc/evaluate_real_embeddings.py [since] [until]

Defaults to the last six months of data.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def main(since: str, until: str, n_clusters: int = 32, ticker: str = "SPY") -> int:
    from newsmood import (
        ingest_searches,
        DEFAULT_SEARCHES_ROOT,
        default_embedding_cache,
        DEFAULT_SEED_PROMPTS,
        build_seed_embeddings,
        panel_with_targets,
        attach_targets,
        fit_news_clusters,
        attach_cluster_id,
        session_cluster_counts,
        evaluate_walk_forward,
    )
    import dol

    t0 = time.time()
    cache = default_embedding_cache()
    print(f"cache size: {len(list(cache)):,}")

    store = dol.JsonFiles(DEFAULT_SEARCHES_ROOT)
    df = ingest_searches(store, source="newsdata", since=since, until=until)
    print(f"ingested: {len(df):,} rows ({time.time()-t0:.1f}s)")

    # Only keep rows whose vectors are already cached
    df["vector"] = df["doc_id"].map(lambda did: cache.get(did) if hasattr(cache, "get") else None)
    if df["vector"].isna().any():
        # MutableMapping without .get
        cached_set = set(cache)
        df = df[df["doc_id"].isin(cached_set)].copy()
        df["vector"] = df["doc_id"].map(lambda did: cache[did])
    df = df.dropna(subset=["vector"]).reset_index(drop=True)
    print(f"with real vectors: {len(df):,} rows")
    if df.empty:
        print("No embedded rows in this window — run embed_dataframe first.")
        return 1

    # Cluster + targets
    print("fitting clusters...")
    model_c = fit_news_clusters(df, n_clusters=n_clusters, random_state=0, sample_size=min(50_000, len(df)))
    df = attach_cluster_id(df, model_c)
    print("attaching targets...")
    df = attach_targets(df, tickers=(ticker,), horizons=(1, 5))

    # Seed embeddings (cached automatically since same store)
    print("building seed embeddings (cached)...")
    seed_vecs = build_seed_embeddings(DEFAULT_SEED_PROMPTS, cache=cache)

    # Build panel
    print("computing per-session features...")
    panel = panel_with_targets(df, seed_vecs=seed_vecs, target_cols=[f"{ticker}_ret_1d", f"{ticker}_ret_5d"])
    cluster_counts = session_cluster_counts(df, n_clusters=n_clusters)
    panel = panel.join(cluster_counts, how="left").fillna(0)
    print(f"panel: {panel.shape}")

    # Walk-forward evaluate
    print()
    for target in (f"{ticker}_ret_1d", f"{ticker}_ret_5d"):
        if target not in panel.columns:
            continue
        res = evaluate_walk_forward(
            panel,
            target=target,
            n_splits=5,
            min_train=max(30, len(panel) // 5),
            embargo=5,
        )
        print(f"--- {target} ---")
        if res.per_fold.empty:
            print("  (insufficient data for walk-forward CV)")
            continue
        print(res.per_fold.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
        print()
        print("mean:")
        print(res.summary().to_string(float_format=lambda v: f"{v: .4f}"))
        print()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2025-04-01")
    parser.add_argument("--until", default="2025-05-01")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--n-clusters", type=int, default=32)
    args = parser.parse_args()
    sys.exit(main(args.since, args.until, n_clusters=args.n_clusters, ticker=args.ticker))
