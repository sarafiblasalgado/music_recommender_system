"""Relevance-threshold robustness check: does the headline model ranking
in REPORT.md S6 depend on the specific choice of "relevant" (a held-out
play count at or above the user's own *median* training play count,
config.RELEVANCE_PERCENTILE=0.5)?

Every ranking metric in this project (precision, recall, NDCG, MRR,
hit-rate) is defined relative to that one threshold choice, picked
because there's no natural rating scale to fall back on for implicit
feedback -- but it was never checked whether a different, still-defensible
choice of threshold would reorder which model looks best. This script
re-evaluates a representative subset of models (the non-personalized
baseline, the strongest model from each paradigm, and both social-layer
variants) against three different thresholds:

  - q=0.00 ("any play counts"): every held-out interaction is relevant
  - q=0.50 (the project's default): "an artist this user actually favoured"
  - q=0.75 ("favourite-tier only"): a stricter bar

Models are fit once each on the standard 80/20 split, reusing the
hyperparameters already validated by main.py's grid search (n_factors=20,
cf_k=30) rather than re-running that search here -- the question this
script answers is about the relevance *definition*, not about re-tuning.
"""

import pandas as pd

from src import config
from src.data_loading import (
    load_interactions, load_artists, load_tags, load_user_tagged_artists, load_friends,
    build_artist_tag_corpus, train_test_split_interactions, get_user_quantile_weights,
)
from src.baselines import MostPopularRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering
from src.matrix_factorization import ImplicitALSRecommender
from src.social_filtering import FriendBasedRecommender, GraphDiffusionRecommender
from src.evaluation import evaluate_model

THRESHOLDS = {
    "any_play (q=0.00)": 0.00,
    "median (q=0.50, REPORT.md default)": 0.50,
    "favourite_tier (q=0.75)": 0.75,
}


def main():
    print("Relevance-threshold robustness check")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    interactions = load_interactions()
    artists = load_artists()
    tags = load_tags()
    user_tagged_artists = load_user_tagged_artists()
    friends = load_friends()
    tag_corpus = build_artist_tag_corpus(user_tagged_artists, tags)

    train, test = train_test_split_interactions(interactions, test_size=0.2)
    eval_users = sorted(test[config.USER_COL].unique())

    print("\nFitting models (reusing main.py's validated hyperparameters)...")
    models = {
        "most_popular": MostPopularRecommender().fit(train),
        "content_based_raw": ContentBasedRecommender(use_tfidf=False).fit(train, artists, tag_corpus),
        "item_item_cf": ItemItemCollaborativeFiltering(k=30).fit(train),
        "matrix_factorization": ImplicitALSRecommender(n_factors=20, n_iterations=15).fit(train),
        "friend_based": FriendBasedRecommender().fit(train, friends),
        "graph_diffusion_social": GraphDiffusionRecommender().fit(train, friends),
    }

    rows = []
    for label, q in THRESHOLDS.items():
        user_weights = get_user_quantile_weights(train, q=q)
        global_weight = float(train[config.WEIGHT_COL].quantile(q))
        print(f"\n--- relevance definition: {label} ---")
        for name, model in models.items():
            res = evaluate_model(model, train, test, eval_users, k=config.TOP_K,
                                  user_median_weights=user_weights, global_median_weight=global_weight)
            p, n = res[f"precision@{config.TOP_K}"], res[f"ndcg@{config.TOP_K}"]
            rows.append({"relevance_definition": label, "model": name,
                         f"precision@{config.TOP_K}": p, f"ndcg@{config.TOP_K}": n})
            print(f"  {name:24s} P@10={p:.4f}  NDCG@10={n:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "relevance_threshold_check.csv", index=False)

    pivot = df.pivot(index="model", columns="relevance_definition", values=f"precision@{config.TOP_K}")
    ranks = pivot.rank(ascending=False)
    print("\nModel ranking by precision@10 (1 = best), per relevance definition:")
    print(ranks.to_string())

    print(f"\nSaved to {config.RESULTS_DIR / 'relevance_threshold_check.csv'}")


if __name__ == "__main__":
    main()
