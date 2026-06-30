"""The social layer: this is the extra 'layer' the plain MovieLens-style
template doesn't have at all -- the HetRec2011 Last.fm dataset ships a
friend graph (25,434 directed / 12,717 mutual friendships) alongside the
listening data, so we can recommend based on *who a user knows*, not just
*what they've played*.

Three recommenders live here:

- `FriendBasedRecommender`: a pure social recommender -- "what are my
  friends listening to that I'm not?" This is a classic trust-based
  recommendation strategy and a sensible fallback for users with too
  little listening history to support content-based or CF profiles, as
  long as they have friends.
- `GraphDiffusionRecommender`: the same idea extended past one hop --
  see its own docstring below.
- `HybridRecommender`: a generic score-fusion wrapper that blends *any*
  number of already-fitted recommenders (e.g. item-item CF + the social
  recommender) using weighted reciprocal-rank fusion. Rank fusion is used
  instead of combining raw scores directly because every model here lives
  on a different, incomparable scale (cosine similarity vs. ALS dot
  product vs. summed log-play-counts) -- converting each model's output to
  a rank-based score first makes the blend meaningful regardless of the
  underlying score's units.
"""

import numpy as np
from scipy.sparse import csr_matrix

from . import config
from .data_loading import get_seen_items


class FriendBasedRecommender:
    """Recommend artists popular among a user's friends, weighted by how
    much each friend listened to them (sum of friends' log-play-counts).
    """

    def __init__(self):
        self.friends_of_ = None       # dict: userID -> set(friendID)
        self._interactions_train = None

    def fit(self, interactions, friends):
        self.friends_of_ = {}
        for user_id, friend_id in zip(friends[config.USER_COL], friends["friendID"]):
            self.friends_of_.setdefault(user_id, set()).add(friend_id)
        return self

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        friend_ids = self.friends_of_.get(user_id, set()) if self.friends_of_ else set()
        if not friend_ids:
            return []  # no social signal available for this user

        friends_plays = interactions_train[interactions_train[config.USER_COL].isin(friend_ids)]
        if len(friends_plays) == 0:
            return []

        scores = friends_plays.groupby(config.ITEM_COL)["log_weight"].sum().sort_values(ascending=False)

        seen = get_seen_items(interactions_train, user_id) if exclude_seen else set()
        recs = [(int(item), float(score)) for item, score in scores.items() if item not in seen]
        return recs[:n]


class GraphDiffusionRecommender:
    """Multi-hop social recommender via Personalized PageRank (Page et al.,
    1999) over the friend graph, instead of FriendBasedRecommender's hard
    one-hop cutoff.

    FriendBasedRecommender only ever looks at a user's *direct* friends --
    a user with one or two friends whose own listening history happens to
    be thin gets a correspondingly thin, noisy signal. This is the same
    local-neighbourhood limitation §8 of REPORT.md diagnoses for item-item
    CF, and the reason Implicit ALS was brought in at all: latent-factor
    models borrow statistical strength across the *whole* matrix instead
    of only directly-overlapping neighbours. This class applies the same
    fix to the social layer instead of the listening layer: every user in
    the network gets a continuous relevance weight to the query user
    (computed by power-iterating a personalized random walk that restarts
    at the query user with probability `restart_prob` at every step) --
    friends score highest, friends-of-friends lower, and so on -- rather
    than a binary one-hop membership test, so a user's effective
    neighbourhood for scoring artists can extend past their direct friend
    list when their direct friends alone don't carry much signal.

    The friend graph is directed in the raw data (`user_friends.dat`) but
    only ~50% mutual (12,717 of 25,434 edges) -- the diffusion graph
    symmetrizes it, since a recorded friendship is informative regardless
    of which direction HetRec happened to log it in.
    """

    def __init__(self, restart_prob=0.15, n_iterations=20, top_neighbors=100):
        self.restart_prob = restart_prob
        self.n_iterations = n_iterations
        self.top_neighbors = top_neighbors
        self.user_ids_ = None
        self.user_id_to_index_ = None
        self.transition_ = None  # row-stochastic sparse transition matrix

    def fit(self, interactions, friends):
        user_ids = np.sort(np.union1d(friends[config.USER_COL].unique(), friends["friendID"].unique()))
        self.user_ids_ = user_ids
        self.user_id_to_index_ = {u: i for i, u in enumerate(user_ids)}
        n = len(user_ids)

        rows = friends[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        cols = friends["friendID"].map(self.user_id_to_index_).to_numpy()
        all_rows = np.concatenate([rows, cols])
        all_cols = np.concatenate([cols, rows])
        adjacency = csr_matrix((np.ones(len(all_rows)), (all_rows, all_cols)), shape=(n, n))
        adjacency.data[:] = 1.0
        adjacency.sum_duplicates()
        adjacency.data[:] = 1.0  # de-duplicate edge weight after symmetrizing + summing

        out_degree = np.asarray(adjacency.sum(axis=1)).ravel()
        out_degree[out_degree == 0] = 1.0
        self.transition_ = adjacency.multiply(1.0 / out_degree[:, None]).tocsr()
        return self

    def _personalized_pagerank(self, user_id):
        idx = self.user_id_to_index_.get(user_id)
        if idx is None:
            return None
        restart = np.zeros(len(self.user_ids_))
        restart[idx] = 1.0
        r = restart.copy()
        for _ in range(self.n_iterations):
            r = (1 - self.restart_prob) * (self.transition_.T @ r) + self.restart_prob * restart
        r[idx] = 0.0  # exclude the query user from their own neighbourhood
        return r

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        ppr = self._personalized_pagerank(user_id)
        if ppr is None or not np.any(ppr):
            return []

        k = min(self.top_neighbors, len(ppr) - 1)
        top_idx = np.argpartition(-ppr, k)[:k]
        top_idx = top_idx[ppr[top_idx] > 0]
        if len(top_idx) == 0:
            return []
        neighbor_weight = dict(zip(self.user_ids_[top_idx], ppr[top_idx]))

        neighbor_plays = interactions_train[interactions_train[config.USER_COL].isin(neighbor_weight)]
        if len(neighbor_plays) == 0:
            return []
        neighbor_plays = neighbor_plays.copy()
        neighbor_plays["_w"] = neighbor_plays[config.USER_COL].map(neighbor_weight) * neighbor_plays["log_weight"]
        scores = neighbor_plays.groupby(config.ITEM_COL)["_w"].sum().sort_values(ascending=False)

        seen = get_seen_items(interactions_train, user_id) if exclude_seen else set()
        recs = [(int(item), float(score)) for item, score in scores.items() if item not in seen]
        return recs[:n]


class HybridRecommender:
    """Generic weighted reciprocal-rank-fusion blend of several fitted
    recommenders.

    For each component model we take its top `pool_size` recommendations
    and convert rank -> a 1/(rank+1) score (reciprocal-rank fusion, the
    same technique search engines use to merge differently-scaled ranked
    lists). Final score for an item is the weighted sum of its
    reciprocal-rank contributions across all the models that recommended
    it; an item only one model surfaced still competes, just with a
    smaller contribution.
    """

    def __init__(self, models, weights=None, pool_size=50):
        """models: dict[name -> already-fitted recommender object].
        weights: dict[name -> float], defaults to equal weighting."""
        self.models = models
        self.weights = weights or {name: 1.0 for name in models}
        self.pool_size = pool_size

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        combined_scores = {}
        any_contributed = False

        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0)
            recs = model.recommend(user_id, interactions_train, n=self.pool_size, exclude_seen=exclude_seen)
            if not recs:
                continue
            any_contributed = True
            for rank, (item_id, _score) in enumerate(recs):
                rr_score = weight / (rank + 1.0)
                combined_scores[item_id] = combined_scores.get(item_id, 0.0) + rr_score

        if not any_contributed:
            return []

        ranked = sorted(combined_scores.items(), key=lambda kv: -kv[1])
        return ranked[:n]
