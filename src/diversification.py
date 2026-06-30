"""Diversity-aware re-ranking: Maximal Marginal Relevance (MMR).

§6/§8 of REPORT.md measure that Implicit ALS -- the most *accurate* model
in this project by a wide margin -- is also one of the most
popularity-biased and least diverse, and flag that as a measured-but-not-
mitigated limitation. This module is the fix: a generic post-hoc
re-ranking wrapper (Carbonell & Goldstein, 1998, "The Use of MMR,
Diversity-Based Reranking for Reordering Documents and Producing
Summaries") that greedily trades a controlled amount of relevance for
less redundancy among the items in one user's list, using the same
content-based tag vectors already used to measure intra-list diversity
(src/evaluation.py) -- so "diversity" means the same thing whether it's
being measured or being optimized for.

This is deliberately a re-ranking layer, not a new base recommender: it
takes any fitted model's candidate list and reorders it, the same
architectural pattern as BackfillRecommender (src/backfill.py) wrapping a
base model rather than reimplementing one.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def mmr_rerank(candidates, item_features, item_id_to_index, k=10, lambda_param=0.5):
    """Greedily re-order `candidates` (a list of (item_id, score), already
    ranked by relevance) to balance relevance against redundancy with
    items already selected:

        MMR(i) = lambda * relevance(i) - (1 - lambda) * max_{j in selected} sim(i, j)

    lambda_param=1.0 reduces to the original relevance order (no
    diversification); lower values increasingly favour items dissimilar
    to what's already been picked. Relevance scores are min-max normalized
    to [0, 1] first since different base models live on incomparable
    scales (ALS dot products vs. cosine similarities vs. play counts).
    Items with no tag vector (rare; see content_based.py) are treated as
    maximally dissimilar to everything already selected, the same
    convention evaluation.intra_list_diversity uses when skipping them.
    """
    if not candidates:
        return []

    item_ids = [item_id for item_id, _ in candidates]
    score_by_item = dict(candidates)
    raw_scores = np.array([score_by_item[i] for i in item_ids], dtype=float)
    lo, hi = raw_scores.min(), raw_scores.max()
    if hi > lo:
        norm_scores = (raw_scores - lo) / (hi - lo)
    else:
        norm_scores = np.full_like(raw_scores, 0.5)
    relevance = dict(zip(item_ids, norm_scores))

    feature_row = {i: item_id_to_index[i] for i in item_ids if i in item_id_to_index}

    selected, selected_rows = [], []
    remaining = list(item_ids)

    while remaining and len(selected) < k:
        best_item, best_mmr = None, -np.inf
        for item_id in remaining:
            if selected_rows and item_id in feature_row:
                sims = cosine_similarity(item_features[feature_row[item_id]], item_features[selected_rows]).ravel()
                max_sim = float(sims.max())
            else:
                max_sim = 0.0
            mmr_score = lambda_param * relevance[item_id] - (1 - lambda_param) * max_sim
            if mmr_score > best_mmr:
                best_mmr, best_item = mmr_score, item_id

        selected.append(best_item)
        if best_item in feature_row:
            selected_rows.append(feature_row[best_item])
        remaining.remove(best_item)

    return [(item_id, score_by_item[item_id]) for item_id in selected]


class MMRRecommender:
    """Wraps any fitted recommender and re-ranks its candidate pool with
    MMR before returning the top n. Pulls a pool larger than n from the
    base model (pool_size) so there's something to diversify *among* --
    re-ranking exactly n candidates with no slack just reorders the same
    n items.
    """

    def __init__(self, base_model, item_features, item_id_to_index, lambda_param=0.5, pool_size=50):
        self.base_model = base_model
        self.item_features = item_features
        self.item_id_to_index = item_id_to_index
        self.lambda_param = lambda_param
        self.pool_size = pool_size

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        pool = self.base_model.recommend(
            user_id, interactions_train, n=max(self.pool_size, n), exclude_seen=exclude_seen
        )
        if not pool:
            return []
        return mmr_rerank(pool, self.item_features, self.item_id_to_index, k=n, lambda_param=self.lambda_param)
