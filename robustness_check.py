"""Multi-seed robustness check: re-run the train/test split, model
fitting, and evaluation across several random seeds and report mean +/-
std for each model's accuracy metrics.

This exists because main.py's headline results table (results/metrics.csv)
is a single 80/20 split with random_state=42 -- a real limitation flagged
in REPORT.md: a single split can't tell you whether a gap between two
models (e.g. ALS's 0.13 vs. Friend-Based's 0.08 precision@10) is a stable
property of the algorithms or an artifact of which 20% of each user's
listening history happened to land in the test set. Run separately from
main.py (not on every pipeline run) because it repeats the most expensive
step -- fitting implicit ALS -- once per seed.

Run with: python3 robustness_check.py
"""

import time

import numpy as np
import pandas as pd

from src import config
from src.data_loading import (
    load_interactions, load_artists, load_tags, load_user_tagged_artists, load_friends,
    build_artist_tag_corpus, train_test_split_interactions, get_user_median_weights,
)
from src.baselines import MostPopularRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering, UserUserCollaborativeFiltering
from src.matrix_factorization import ImplicitALSRecommender
from src.social_filtering import FriendBasedRecommender, HybridRecommender
from src.evaluation import evaluate_model

SEEDS = (42, 1, 7, 123, 2024)
MODEL_NAMES = ["most_popular", "content_based", "user_user_cf", "item_item_cf",
               "matrix_factorization", "friend_based", "hybrid_cf_social"]
METRICS = ["precision@10", "recall@10", "ndcg@10", "mrr@10", "hit_rate@10"]


def fit_and_evaluate_one_seed(interactions, artists, tag_corpus, friends, seed):
    train, test = train_test_split_interactions(interactions, test_size=0.2, random_state=seed)
    eval_users = sorted(test[config.USER_COL].unique())
    user_median_weights = get_user_median_weights(train)
    global_median_weight = float(train[config.WEIGHT_COL].median())

    models = {
        "most_popular": MostPopularRecommender().fit(train),
        "content_based": ContentBasedRecommender().fit(train, artists, tag_corpus),
        "user_user_cf": UserUserCollaborativeFiltering(k=20).fit(train),
        "item_item_cf": ItemItemCollaborativeFiltering(k=20).fit(train),
        "matrix_factorization": ImplicitALSRecommender(n_factors=50, n_iterations=15, random_state=seed).fit(train),
        "friend_based": FriendBasedRecommender().fit(train, friends),
    }
    # Fixed at the weight tune_hybrid_weight() validated in main.py (see
    # REPORT.md S6) -- re-tuning per seed here would conflate "is the
    # algorithm stable" with "is the tuning procedure stable", a separate
    # question this check isn't trying to answer.
    models["hybrid_cf_social"] = HybridRecommender(
        models={"item_item_cf": models["item_item_cf"], "friend_based": models["friend_based"]},
        weights={"item_item_cf": 0.0, "friend_based": 1.0},
        pool_size=50,
    )

    rows = {}
    for name in MODEL_NAMES:
        res = evaluate_model(models[name], train, test, eval_users, k=config.TOP_K,
                              user_median_weights=user_median_weights, global_median_weight=global_median_weight)
        rows[name] = {m: res[m] for m in METRICS}
    return rows


def main():
    print(f"Robustness check: {len(SEEDS)} seeds {SEEDS}, {len(MODEL_NAMES)} models")
    interactions = load_interactions()
    artists = load_artists()
    tags = load_tags()
    user_tagged_artists = load_user_tagged_artists()
    friends = load_friends()
    tag_corpus = build_artist_tag_corpus(user_tagged_artists, tags)

    per_seed_rows = []
    for seed in SEEDS:
        t0 = time.time()
        seed_results = fit_and_evaluate_one_seed(interactions, artists, tag_corpus, friends, seed)
        for name, metrics in seed_results.items():
            per_seed_rows.append({"seed": seed, "model": name, **metrics})
        print(f"  seed={seed}  done in {time.time() - t0:.1f}s")

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(config.RESULTS_DIR / "metrics_multiseed_raw.csv", index=False)

    summary = per_seed_df.groupby("model")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reindex(MODEL_NAMES)
    summary.to_csv(config.RESULTS_DIR / "metrics_multiseed_summary.csv")

    print("\nMean +/- std across seeds:")
    for name in MODEL_NAMES:
        row = summary.loc[name]
        print(f"  {name:22s} " + "  ".join(
            f"{m}={row[f'{m}_mean']:.4f}+/-{row[f'{m}_std']:.4f}" for m in METRICS))

    print(f"\nSaved per-seed raw results to {config.RESULTS_DIR / 'metrics_multiseed_raw.csv'}")
    print(f"Saved mean/std summary to {config.RESULTS_DIR / 'metrics_multiseed_summary.csv'}")


if __name__ == "__main__":
    main()
