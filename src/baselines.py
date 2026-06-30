"""Non-personalized baseline recommenders.

Every recommender in this project (baselines, content-based, CF, matrix
factorization, social) shares the same `recommend()` contract: a list of
(artist_id, score) tuples, length <= n, sorted by score descending, with
already-listened-to artists excluded if exclude_seen=True. This lets
evaluation.py and main.py treat every model interchangeably.
"""

import numpy as np

from . import config
from .data_loading import get_seen_items


class MostPopularRecommender:
    """Recommend the artists with the most unique listeners.

    We rank by *listener count*, not total plays, deliberately: total
    plays would let a handful of obsessive listeners of a niche artist
    outweigh an artist that many people listen to moderately. Listener
    count is the implicit-feedback analogue of "most rated" in an explicit
    setting.
    """

    def __init__(self):
        self.ranking_ = None

    def fit(self, interactions, artists=None):
        counts = interactions.groupby(config.ITEM_COL).size().sort_values(ascending=False)
        self.ranking_ = list(zip(counts.index.tolist(), counts.values.tolist()))
        return self

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        if self.ranking_ is None:
            raise RuntimeError("Call fit() before recommend().")
        seen = get_seen_items(interactions_train, user_id) if exclude_seen else set()
        recs = [(item, float(score)) for item, score in self.ranking_ if item not in seen]
        return recs[:n]


class MostPlayedRecommender:
    """Recommend the artists with the highest total play count, subject to
    a minimum number of listeners.

    This is the implicit-feedback analogue of "highest average rating with
    a minimum number of ratings": without the listener-count floor, a
    single user who looped one obscure track 50,000 times would dominate
    the ranking.
    """

    def __init__(self, min_listeners=20):
        self.min_listeners = min_listeners
        self.ranking_ = None

    def fit(self, interactions, artists=None):
        agg = interactions.groupby(config.ITEM_COL).agg(
            total_plays=(config.WEIGHT_COL, "sum"),
            n_listeners=(config.ITEM_COL, "size"),
        )
        agg = agg[agg["n_listeners"] >= self.min_listeners]
        agg = agg.sort_values("total_plays", ascending=False)
        self.ranking_ = list(zip(agg.index.tolist(), agg["total_plays"].values.tolist()))
        return self

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        if self.ranking_ is None:
            raise RuntimeError("Call fit() before recommend().")
        seen = get_seen_items(interactions_train, user_id) if exclude_seen else set()
        recs = [(item, float(score)) for item, score in self.ranking_ if item not in seen]
        return recs[:n]


class RandomRecommender:
    """Recommend random unseen artists -- the sanity-check lower bound."""

    def __init__(self, random_state=config.RANDOM_STATE):
        self.random_state = random_state
        self.items_ = None

    def fit(self, interactions, artists=None):
        self.items_ = interactions[config.ITEM_COL].unique().tolist()
        return self

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        if self.items_ is None:
            raise RuntimeError("Call fit() before recommend().")
        seen = get_seen_items(interactions_train, user_id) if exclude_seen else set()
        candidates = [i for i in self.items_ if i not in seen]
        rng = np.random.RandomState(self.random_state + int(user_id))
        chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        return [(int(item), float(n - rank)) for rank, item in enumerate(chosen)]
