"""Temporal split check: does a "predict the future from the past" split
change the headline accuracy numbers, compared to the random per-user
split used everywhere else in this project?

Why this needs its own script instead of just adding `split="temporal"` to
main.py: `user_artists.dat` (the play-count data every model trains on)
carries **no timestamp at all**. The only timestamp in the whole dataset
is on tag-application events (`user_taggedartists.dat`), and only ~22% of
(user, artist) listening interactions even have a tag, so an "earliest tag
date" timestamp proxy is only available for a minority subset (see
`data_loading.get_interaction_timestamps`). A true temporal evaluation
across the *entire* dataset is not something this dataset supports --
that's a real, irreducible limitation (see REPORT.md S10), not an
implementation gap.

What this script does instead, to get a defensible answer anyway: restrict
the *entire* comparison -- training data and test data, for *both* split
methods -- to the timestamped subset only (~20,665 rows). That way "random
split" and "temporal split" differ only in *which* interactions get held
out for a user, not in how much data the models see overall, so the
resulting gap (if any) is attributable to split methodology, not to a
difference in data volume. The trade-off: because the model sees ~5x less
data than in the main pipeline, these accuracy numbers are *not*
comparable in absolute terms to results/metrics.csv -- only the relative
gap between the two rows of results/temporal_split_check.csv is
meaningful.

Run with: python3 temporal_split_check.py
"""

from src import config
from src.data_loading import (
    load_interactions, load_artists, load_tags, load_user_tagged_artists,
    get_interaction_timestamps, train_test_split_interactions,
    train_test_split_interactions_temporal, get_user_median_weights,
)
from src.baselines import MostPopularRecommender
from src.matrix_factorization import ImplicitALSRecommender
from src.evaluation import evaluate_model


def restrict_to_timestamped(interactions, timestamps):
    ts_df = timestamps.rename("yyyymm").reset_index()
    return interactions.merge(ts_df, on=[config.USER_COL, config.ITEM_COL], how="inner")[interactions.columns]


def fit_and_evaluate(train, test, label):
    eval_users = sorted(test[config.USER_COL].unique())
    user_median_weights = get_user_median_weights(train)
    global_median_weight = float(train[config.WEIGHT_COL].median())

    most_popular = MostPopularRecommender().fit(train)
    als = ImplicitALSRecommender(n_factors=20, n_iterations=15).fit(train)

    rows = []
    for name, model in [("most_popular", most_popular), ("matrix_factorization", als)]:
        res = evaluate_model(model, train, test, eval_users, k=config.TOP_K,
                              user_median_weights=user_median_weights,
                              global_median_weight=global_median_weight)
        res["model"] = name
        res["split"] = label
        rows.append(res)
        print(f"  [{label:9s}] {name:22s} P@10={res[f'precision@{config.TOP_K}']:.4f}  "
              f"R@10={res[f'recall@{config.TOP_K}']:.4f}  NDCG@10={res[f'ndcg@{config.TOP_K}']:.4f}")
    return rows


def main():
    print("Temporal split check (see module docstring for why this is a separate, scoped script)")
    interactions = load_interactions()
    artists = load_artists()
    tags = load_tags()
    user_tagged_artists = load_user_tagged_artists()

    timestamps = get_interaction_timestamps(user_tagged_artists)
    timestamped = restrict_to_timestamped(interactions, timestamps)
    print(f"\nTotal interactions: {len(interactions):,}")
    print(f"With a tag-timestamp proxy: {len(timestamped):,} "
          f"({len(timestamped) / len(interactions):.1%})")
    print(f"Users with >=1 timestamped interaction: {timestamped[config.USER_COL].nunique():,}")

    print("\nFitting on the random split (restricted to the timestamped subset)...")
    random_train, random_test = train_test_split_interactions(timestamped, test_size=0.2)
    random_rows = fit_and_evaluate(random_train, random_test, "random")

    print("\nFitting on the temporal split (restricted to the timestamped subset)...")
    temporal_train, temporal_test = train_test_split_interactions_temporal(timestamped, timestamps, test_size=0.2)
    temporal_rows = fit_and_evaluate(temporal_train, temporal_test, "temporal")

    import pandas as pd
    df = pd.DataFrame(random_rows + temporal_rows).set_index(["split", "model"])
    cols = [f"precision@{config.TOP_K}", f"recall@{config.TOP_K}", f"ndcg@{config.TOP_K}",
            f"mrr@{config.TOP_K}", f"hit_rate@{config.TOP_K}", "n_users_evaluated"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(config.RESULTS_DIR / "temporal_split_check.csv")

    print("\nResult (NOT comparable in magnitude to results/metrics.csv -- see module docstring):")
    print(df.round(4).to_string())
    print(f"\nSaved to {config.RESULTS_DIR / 'temporal_split_check.csv'}")


if __name__ == "__main__":
    main()
