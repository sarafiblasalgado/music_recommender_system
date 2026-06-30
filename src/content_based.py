"""Content-based recommender for the music domain.

Artists are represented as TF-IDF vectors over the *folksonomy tags*
Last.fm users have applied to them (e.g. "german_metal", "90s",
"female_vocalists") -- this dataset's analogue of MovieLens genres, but
richer: tags are user-generated and far more numerous/specific than a
fixed genre list. A user profile is the listening-weighted centroid of the
artists they've played, and recommendations are unseen artists whose tag
vectors are most cosine-similar to that profile.

Design choices:
- TF-IDF down-weights tags that are common across the whole catalog
  (e.g. "rock") relative to tags that are distinctive (e.g. "riot_grrrl"),
  pulling the user profile toward what's specific about their taste.
- Because play counts have no natural "good/bad" scale (unlike a 1-5 star
  rating, more plays is never literally bad), we center each user's
  log-play-counts by their own mean log-play-count before building the
  profile -- analogous to the explicit-rating centering, but applied to
  listening intensity rather than rating value: artists played *more than
  usual for this user* push the profile toward their tags; artists played
  *less than usual* push it away.
"""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from .data_loading import get_seen_items


class ContentBasedRecommender:
    """Content-based recommender using Last.fm folksonomy tags.

    use_tfidf=True (default) weights tags by TF-IDF, down-weighting tags
    that are common across the whole catalog (e.g. "rock") relative to
    distinctive ones (e.g. "riot_grrrl"). use_tfidf=False instead uses raw
    tag-application counts per artist (a CountVectorizer) -- the simpler
    alternative the assignment explicitly suggests comparing against.

    Counter-intuitively, raw counts *beat* TF-IDF on every accuracy metric
    here (see REPORT.md S4.2 for the full numbers and discussion) -- the
    plausible explanation is that for this dataset, the "common" tags
    TF-IDF down-weights (genre labels like "rock"/"electronic") are
    genuinely load-bearing taste signal, not noise, while up-weighting
    rare tags surfaces idiosyncratic one-off tag applications that don't
    generalize. TF-IDF does win on coverage and novelty, i.e. it's the
    more *exploratory* of the two, just not the more *accurate* one. This
    is a useful reminder that a standard NLP weighting trick doesn't
    automatically transfer to a domain where "common" doesn't mean "less
    informative".
    """

    def __init__(self, use_tfidf=True):
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.item_features_ = None
        self.item_ids_ = None
        self.item_id_to_index_ = None
        self.item_sim_ = None
        self._user_mean_cache = {}

    def fit(self, interactions, artists, tag_corpus):
        """tag_corpus: Series indexed by artistID -> space-separated tag
        tokens (see data_loading.build_artist_tag_corpus)."""
        item_ids = artists[config.ITEM_COL].to_numpy()
        texts = tag_corpus.reindex(item_ids).fillna("")

        VectorizerCls = TfidfVectorizer if self.use_tfidf else CountVectorizer
        self.vectorizer = VectorizerCls(token_pattern=r"[^\s]+", min_df=2)
        self.item_features_ = self.vectorizer.fit_transform(texts)
        self.item_ids_ = item_ids
        self.item_id_to_index_ = {iid: idx for idx, iid in enumerate(item_ids)}
        self.item_sim_ = None
        self._user_mean_cache = {}
        return self

    def build_user_profile(self, user_id, interactions_train):
        """profile(u) = sum_i (log_weight(u,i) - mean_log_weight(u)) * vector(i)."""
        if self.item_features_ is None:
            raise RuntimeError("Call fit() before build_user_profile().")

        user_rows = interactions_train[interactions_train[config.USER_COL] == user_id]
        if len(user_rows) == 0:
            return None

        if user_id not in self._user_mean_cache:
            self._user_mean_cache[user_id] = user_rows["log_weight"].mean()
        mean_lw = self._user_mean_cache[user_id]

        rows, weights = [], []
        for item_id, log_w in zip(user_rows[config.ITEM_COL], user_rows["log_weight"]):
            idx = self.item_id_to_index_.get(item_id)
            if idx is None:
                continue
            rows.append(idx)
            weights.append(log_w - mean_lw)

        if not rows:
            return None

        weights = np.array(weights)
        sub_matrix = self.item_features_[rows]
        profile = sub_matrix.T.dot(weights)
        return np.asarray(profile).ravel()

    def recommend(self, user_id, interactions_train, n=10, exclude_seen=True):
        profile = self.build_user_profile(user_id, interactions_train)
        if profile is None or not np.any(profile):
            return []

        sims = cosine_similarity(profile.reshape(1, -1), self.item_features_).ravel()
        seen = get_seen_items(interactions_train, user_id) if exclude_seen else set()
        order = np.argsort(-sims)

        recs = []
        for idx in order:
            item_id = int(self.item_ids_[idx])
            if item_id in seen:
                continue
            score = float(sims[idx])
            if score <= 0:
                break
            recs.append((item_id, score))
            if len(recs) >= n:
                break
        return recs

    def similar_items(self, item_id, n=10):
        """Artists with the most similar tag profile to a given artist."""
        if self.item_features_ is None:
            raise RuntimeError("Call fit() before similar_items().")
        idx = self.item_id_to_index_.get(item_id)
        if idx is None:
            return []

        if self.item_sim_ is None:
            self.item_sim_ = cosine_similarity(self.item_features_)

        sims = self.item_sim_[idx].copy()
        sims[idx] = -1
        order = np.argsort(-sims)[:n]
        return [(int(self.item_ids_[j]), float(sims[j])) for j in order if sims[j] > 0]
