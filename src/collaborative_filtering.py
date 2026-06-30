"""Collaborative filtering recommenders for implicit listening data.

Design choices, and how they differ from the explicit-rating (MovieLens)
version of this same algorithm:

- **No mean-centering before similarity.** Explicit ratings benefit from
  "adjusted cosine" (subtracting each user's mean rating) because a 3-star
  rating means different things for a generous vs. a harsh rater. Implicit
  play counts don't have that problem the same way: an entry only exists
  because the user *did* listen, so there's no "harsh listener" giving
  artists low play counts on purpose. Subtracting a mean here would turn
  "listened less than average" into a fabricated *negative* signal, which
  is not what zero-or-low plays actually means for unobserved data. We
  therefore compute similarity directly on log1p(play count).
- **Log compression.** Without it, similarity would be dominated entirely
  by the handful of artists a power-listener played tens of thousands of
  times; see the EDA's long-tail play-count distribution.
- **Minimum-support filtering, two layers of it.** Artists with very few
  listeners (or users with very few interactions, for the user-user
  variant) produce similarity estimates with almost no statistical
  support, so we (1) restrict the CF universe to items/users above a
  listener-count floor, and (2) on top of that, zero out any individual
  similarity pair backed by fewer than `min_co_count` shared
  listeners/artists, regardless of how high the raw cosine value is --
  see `ItemItemCollaborativeFiltering.fit` for why this second floor is
  necessary and not just a refinement of the first.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data_loading import get_seen_items


class ItemItemCollaborativeFiltering:
    """Item-item collaborative filtering over log-compressed play counts.

    Significance-weighted ("shrunk") similarity. Raw cosine similarity on
    sparse implicit data has a serious long-tail failure mode: two niche
    artists with only 5-10 listeners each can show cosine similarity near
    1.0 just because the handful of people who listened to both happened
    to play them a lot -- there's no statistical support behind that
    number, but the *raw value* looks identical to a similarity backed by
    hundreds of shared listeners. Left uncorrected, this completely
    swamps recommendations with obscure, spuriously-"similar" artists.
    We apply the standard fix from Sarwar et al.: multiply each similarity
    by co_count / (co_count + beta), where co_count is the number of users
    who interacted with *both* items. This shrinks similarities with weak
    support toward zero without touching well-supported ones.
    """

    def __init__(self, k=20, min_listeners=10, shrinkage_beta=10, min_co_count=15):
        self.k = k
        self.min_listeners = min_listeners
        self.shrinkage_beta = shrinkage_beta
        self.min_co_count = min_co_count

        self.user_item_matrix_ = None   # log1p(weight), sparse, shape (n_users, n_items)
        self.item_similarity_ = None    # dense, shape (n_items, n_items), shrinkage-corrected
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def fit(self, interactions):
        item_counts = interactions.groupby(config.ITEM_COL).size()
        keep_items = item_counts[item_counts >= self.min_listeners].index
        df = interactions[interactions[config.ITEM_COL].isin(keep_items)]

        self.user_ids_ = np.sort(df[config.USER_COL].unique())
        self.item_ids_ = np.sort(df[config.ITEM_COL].unique())
        self.user_id_to_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_id_to_index_ = {m: i for i, m in enumerate(self.item_ids_)}

        rows = df[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        cols = df[config.ITEM_COL].map(self.item_id_to_index_).to_numpy()
        vals = df["log_weight"].to_numpy()

        self.user_item_matrix_ = csr_matrix(
            (vals, (rows, cols)), shape=(len(self.user_ids_), len(self.item_ids_))
        )
        sim = cosine_similarity(self.user_item_matrix_.T)

        binary = csr_matrix((np.ones_like(vals), (rows, cols)), shape=self.user_item_matrix_.shape)
        co_count = (binary.T @ binary).toarray()  # shared-listener counts, item x item
        shrinkage = co_count / (co_count + self.shrinkage_beta)

        sim = sim * shrinkage
        # Hard floor: a pair backed by fewer than `min_co_count` shared
        # listeners is dropped entirely (set to exactly 0), not just
        # shrunk. This matters because the prediction formula is a
        # *weighted average* -- if only one neighbor passes the top-k cut,
        # its similarity magnitude cancels out of the ratio entirely and
        # the prediction silently collapses to "whatever the user rated
        # that one neighbor", however unreliable the single similarity
        # value was. The hard floor prevents near-zero-support pairs from
        # ever being treated as a real neighbor in the first place.
        sim[co_count < self.min_co_count] = 0.0

        self.item_similarity_ = sim
        np.fill_diagonal(self.item_similarity_, 0.0)
        return self

    def predict_score(self, user_id, item_id):
        """score(u,i) = sum_j sim(i,j) * logplay(u,j) / sum_j |sim(i,j)|,
        over the user's top-k most-similar already-played artists j."""
        if user_id not in self.user_id_to_index_ or item_id not in self.item_id_to_index_:
            return 0.0

        u_idx = self.user_id_to_index_[user_id]
        i_idx = self.item_id_to_index_[item_id]

        user_row = self.user_item_matrix_.getrow(u_idx)
        played_idx, played_vals = user_row.indices, user_row.data
        if len(played_idx) == 0:
            return 0.0

        sims = self.item_similarity_[i_idx, played_idx]
        if self.k < len(sims):
            top_k = np.argpartition(-np.abs(sims), self.k)[: self.k]
            sims, played_vals = sims[top_k], played_vals[top_k]

        denom = np.sum(np.abs(sims))
        if denom < 1e-8:
            return 0.0
        return float(np.dot(sims, played_vals) / denom)

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        if self.item_similarity_ is None:
            raise RuntimeError("Call fit() before recommend().")
        if user_id not in self.user_id_to_index_:
            return []

        seen = get_seen_items(interactions_train, user_id) if exclude_seen else set()
        u_idx = self.user_id_to_index_[user_id]
        user_row = self.user_item_matrix_.getrow(u_idx)
        played_idx, played_vals = user_row.indices, user_row.data
        if len(played_idx) == 0:
            return []

        sims_block = self.item_similarity_[:, played_idx]  # (n_items, n_played)
        if self.k < len(played_idx):
            top_k_idx = np.argpartition(-np.abs(sims_block), self.k, axis=1)[:, : self.k]
            row_idx = np.arange(sims_block.shape[0])[:, None]
            sims_topk = sims_block[row_idx, top_k_idx]
            vals_topk = played_vals[top_k_idx]
        else:
            sims_topk = sims_block
            vals_topk = np.broadcast_to(played_vals, sims_block.shape)

        numer = np.sum(sims_topk * vals_topk, axis=1)
        denom = np.sum(np.abs(sims_topk), axis=1)
        scores = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 1e-8)

        order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
        recs = []
        for idx in order:
            item_id = int(self.item_ids_[idx])
            if item_id in seen or np.isnan(scores[idx]):
                continue
            recs.append((item_id, float(scores[idx])))
            if len(recs) >= n:
                break
        return recs


class UserUserCollaborativeFiltering:
    """User-user collaborative filtering: predict from similar users' listening.

    Uses the same significance-weighted similarity correction as the
    item-item variant (see its docstring): two users who happen to share
    only 2-3 played artists can otherwise show spuriously high cosine
    similarity.
    """

    def __init__(self, k=20, min_interactions=config.MIN_INTERACTIONS_PER_USER, shrinkage_beta=10, min_co_count=8):
        self.k = k
        self.min_interactions = min_interactions
        self.shrinkage_beta = shrinkage_beta
        self.min_co_count = min_co_count

        self.user_item_matrix_ = None
        self.user_similarity_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def fit(self, interactions):
        user_counts = interactions.groupby(config.USER_COL).size()
        keep_users = user_counts[user_counts >= self.min_interactions].index
        df = interactions[interactions[config.USER_COL].isin(keep_users)]

        self.user_ids_ = np.sort(df[config.USER_COL].unique())
        self.item_ids_ = np.sort(df[config.ITEM_COL].unique())
        self.user_id_to_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_id_to_index_ = {m: i for i, m in enumerate(self.item_ids_)}

        rows = df[config.USER_COL].map(self.user_id_to_index_).to_numpy()
        cols = df[config.ITEM_COL].map(self.item_id_to_index_).to_numpy()
        vals = df["log_weight"].to_numpy()

        self.user_item_matrix_ = csr_matrix(
            (vals, (rows, cols)), shape=(len(self.user_ids_), len(self.item_ids_))
        )
        sim = cosine_similarity(self.user_item_matrix_)

        binary = csr_matrix((np.ones_like(vals), (rows, cols)), shape=self.user_item_matrix_.shape)
        co_count = (binary @ binary.T).toarray()  # shared-artist counts, user x user
        shrinkage = co_count / (co_count + self.shrinkage_beta)

        sim = sim * shrinkage
        sim[co_count < self.min_co_count] = 0.0  # same hard floor rationale as item-item CF

        self.user_similarity_ = sim
        np.fill_diagonal(self.user_similarity_, 0.0)
        return self

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        if self.user_similarity_ is None:
            raise RuntimeError("Call fit() before recommend().")
        if user_id not in self.user_id_to_index_:
            return []

        u_idx = self.user_id_to_index_[user_id]
        sims = self.user_similarity_[u_idx]
        if self.k < len(sims):
            top_k_users = np.argpartition(-np.abs(sims), self.k)[: self.k]
        else:
            top_k_users = np.arange(len(sims))

        neighbor_sims = sims[top_k_users]
        neighbor_vals = self.user_item_matrix_[top_k_users].toarray()
        mask = neighbor_vals > 0

        numer = (neighbor_sims[:, None] * neighbor_vals).sum(axis=0)
        denom = (np.abs(neighbor_sims)[:, None] * mask).sum(axis=0)
        scores = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=denom > 1e-8)

        seen = get_seen_items(interactions_train, user_id) if exclude_seen else set()
        order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))

        recs = []
        for idx in order:
            item_id = int(self.item_ids_[idx])
            if item_id in seen or np.isnan(scores[idx]):
                continue
            recs.append((item_id, float(scores[idx])))
            if len(recs) >= n:
                break
        return recs
