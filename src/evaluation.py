"""Evaluation metrics for the music recommender.

Ranking metrics (Precision/Recall/NDCG/MRR/Hit-Rate) use binary relevance,
same as the explicit-rating case, but relevance itself is defined
implicitly (see data_loading.get_relevant_items): a held-out interaction
counts as relevant if its play count is at or above the user's own median
training play count. Beyond-accuracy metrics (catalog coverage, novelty,
diversity) matter even more here than for movies, since a music
recommender that only ever resurfaces global top-40 artists is a much
more obviously bad product experience than one that does the analogous
thing for films. The three beyond-accuracy metrics answer different
questions: coverage asks "how much of the catalog gets recommended to
*anyone*", novelty asks "how non-obvious are the items in a typical
list", and diversity asks "how different are the items *within one
list* from each other".
"""

import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data_loading import get_relevant_items


def precision_at_k(recommended_items, relevant_items, k=10):
    if k <= 0:
        return 0.0
    top_k = list(recommended_items)[:k]
    if not top_k:
        return 0.0
    hits = len(set(top_k) & set(relevant_items))
    return hits / len(top_k)


def recall_at_k(recommended_items, relevant_items, k=10):
    if not relevant_items:
        return 0.0
    top_k = list(recommended_items)[:k]
    hits = len(set(top_k) & set(relevant_items))
    return hits / len(relevant_items)


def hit_rate_at_k(recommended_items, relevant_items, k=10):
    top_k = set(list(recommended_items)[:k])
    return 1.0 if top_k & set(relevant_items) else 0.0


def dcg_at_k(relevance_scores, k=10):
    scores = np.asarray(list(relevance_scores)[:k], dtype=float)
    if scores.size == 0:
        return 0.0
    ranks = np.arange(1, scores.size + 1)
    return float(np.sum(scores / np.log2(ranks + 1)))


def ndcg_at_k(recommended_items, relevant_items, k=10):
    top_k = list(recommended_items)[:k]
    relevant_items = set(relevant_items)
    rel = [1.0 if item in relevant_items else 0.0 for item in top_k]

    dcg = dcg_at_k(rel, k)
    idcg = dcg_at_k([1.0] * min(k, len(relevant_items)), k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mean_reciprocal_rank(recommended_items, relevant_items, k=10):
    relevant_items = set(relevant_items)
    for rank, item in enumerate(list(recommended_items)[:k], start=1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


def catalog_coverage(all_recommendations, all_items):
    all_items = set(all_items)
    if not all_items:
        return 0.0
    recommended = set(all_recommendations)
    return len(recommended & all_items) / len(all_items)


def novelty_at_k(recommended_items, item_popularity, n_users, k=10):
    """Mean self-information (bits) of the top-k recommended artists:
    -log2(P(artist)), P(artist) = (#listeners) / n_users."""
    top_k = list(recommended_items)[:k]
    if not top_k:
        return 0.0
    vals = []
    for item in top_k:
        count = item_popularity.get(item, 1)
        p = max(count / n_users, 1e-12)
        vals.append(-np.log2(p))
    return float(np.mean(vals))


def compute_popularity_percentiles(item_popularity):
    """Map each item to its percentile rank (in (0, 1], 1.0 = the single
    most popular item) within the training listener-count distribution.
    Used by popularity_bias() -- a direct, easily-interpreted "how much of
    the catalog's head does this list lean on" measure, distinct from
    novelty (which is in bits of self-information) and catalog coverage
    (which is about spread *across users*, not concentration *within one
    list*)."""
    items = list(item_popularity.keys())
    counts = np.array([item_popularity[i] for i in items], dtype=float)
    order = np.argsort(counts)
    ranks = np.empty(len(counts))
    ranks[order] = np.arange(1, len(counts) + 1) / len(counts)
    return dict(zip(items, ranks.tolist()))


def popularity_bias(recommended_items, item_popularity_percentile, k=10):
    """Mean popularity percentile of the top-k recommended items (0 =
    only ever recommends the most obscure items in the catalog, 1 = only
    ever recommends the single most popular item). A model that scores
    well on novelty/coverage in aggregate can still be popularity-biased
    *within* a given list if it leans on a handful of moderately popular
    artists every time -- this metric is sensitive to that in a way
    novelty's log-scale and coverage's across-user view aren't."""
    top_k = list(recommended_items)[:k]
    if not top_k:
        return None
    vals = [item_popularity_percentile.get(item, 0.0) for item in top_k]
    return float(np.mean(vals))


def intra_list_diversity(recommended_items, item_features, item_id_to_index, k=10):
    """Mean pairwise dissimilarity (1 - cosine similarity) between a user's
    top-k recommended items' content (tag) vectors -- the standard
    beyond-accuracy diversity metric, distinct from coverage (which is
    about spread *across users*) and novelty (which is about absolute
    popularity, not how similar the items in one list are *to each
    other*). A model that always recommends ten variations on the same
    sub-genre scores low here even if its accuracy and novelty look fine.
    Uses the content-based recommender's TF-IDF tag vectors so diversity
    is measured the same way regardless of which algorithm produced the
    list. Items without a tag vector are skipped; if fewer than two
    recommended items have one, diversity is undefined for that user.
    """
    top_k = list(recommended_items)[:k]
    idx = [item_id_to_index[i] for i in top_k if i in item_id_to_index]
    if len(idx) < 2:
        return None
    sims = cosine_similarity(item_features[idx])
    iu = np.triu_indices(sims.shape[0], k=1)
    return float(1.0 - np.mean(sims[iu]))


def evaluate_model(model, interactions_train, interactions_test, users, k=10,
                    user_median_weights=None, global_median_weight=None,
                    item_popularity=None, n_users=None, all_items=None,
                    item_features=None, item_id_to_index=None,
                    item_popularity_percentile=None, return_per_user=False):
    """Evaluate a recommender over a set of users, exactly mirroring the
    explicit-rating evaluation loop but with implicit relevance.

    return_per_user=True additionally returns the raw per-evaluated-user
    precision/NDCG arrays (aligned to each other, one entry per evaluable
    user) under "per_user_precision"/"per_user_ndcg" -- used by
    run_significance_tests() for paired statistical tests between models,
    which need the same users' scores side by side rather than just the
    aggregate mean.
    """
    precisions, recalls, ndcgs, mrrs, hit_rates = [], [], [], [], []
    novelties, diversities, pop_biases = [], [], []
    all_recs = []
    n_evaluable = 0
    n_no_recs = 0

    for user_id in users:
        recs = model.recommend(user_id, interactions_train, n=k, exclude_seen=True)
        rec_items = [item for item, _ in recs]
        all_recs.extend(rec_items)

        if not rec_items:
            n_no_recs += 1

        if item_popularity is not None and n_users is not None and rec_items:
            novelties.append(novelty_at_k(rec_items, item_popularity, n_users, k))

        if item_features is not None and item_id_to_index is not None:
            diversity = intra_list_diversity(rec_items, item_features, item_id_to_index, k)
            if diversity is not None:
                diversities.append(diversity)

        if item_popularity_percentile is not None and rec_items:
            bias = popularity_bias(rec_items, item_popularity_percentile, k)
            if bias is not None:
                pop_biases.append(bias)

        relevant = get_relevant_items(interactions_test, user_id, user_median_weights, global_median_weight)
        if not relevant:
            continue
        n_evaluable += 1

        precisions.append(precision_at_k(rec_items, relevant, k))
        recalls.append(recall_at_k(rec_items, relevant, k))
        ndcgs.append(ndcg_at_k(rec_items, relevant, k))
        mrrs.append(mean_reciprocal_rank(rec_items, relevant, k))
        hit_rates.append(hit_rate_at_k(rec_items, relevant, k))

    result = {
        "k": k,
        "n_users_evaluated": n_evaluable,
        "n_users_total": len(users),
        "n_users_no_recs": n_no_recs,
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"mrr@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"hit_rate@{k}": float(np.mean(hit_rates)) if hit_rates else 0.0,
    }
    if all_items is not None:
        result["catalog_coverage"] = catalog_coverage(all_recs, all_items)
    if novelties:
        result["novelty"] = float(np.mean(novelties))
    if diversities:
        result["diversity"] = float(np.mean(diversities))
    if pop_biases:
        result["popularity_bias"] = float(np.mean(pop_biases))
    if return_per_user:
        result["per_user_precision"] = precisions
        result["per_user_ndcg"] = ndcgs
    return result


def paired_wilcoxon_test(scores_a, scores_b):
    """Wilcoxon signed-rank test on two models' per-user metric scores
    (e.g. each model's per_user_ndcg from evaluate_model(..., return_per_user=True),
    for the *same* users in the *same* order). Used instead of a paired
    t-test because per-user precision/NDCG is bounded in [0, 1] and
    heavily zero-inflated (most users get 0 on most models at k=10) --
    clearly not normally distributed, so a rank-based test is the
    appropriate one. Returns (statistic, p_value); if every paired
    difference is exactly zero (the two models produced identical
    rankings for every user), the test is undefined and this returns
    (0.0, 1.0) rather than raising.
    """
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    if len(a) != len(b):
        raise ValueError("paired_wilcoxon_test requires equal-length, aligned score arrays")
    if np.allclose(a, b):
        return 0.0, 1.0
    statistic, p_value = stats.wilcoxon(a, b)
    return float(statistic), float(p_value)
