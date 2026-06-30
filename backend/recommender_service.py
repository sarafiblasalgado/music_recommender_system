"""Recommendation logic for the consumer-facing API.

This is deliberately separate from main.py's offline evaluation pipeline:
that pipeline trains on an 80% split to hold out a test set for measuring
accuracy (see REPORT.md). This service exists to serve a live product, so
it trains every model on the *full* interaction history -- there's no
"test set" to hold back from a real user-facing feed, the goal is simply
the best recommendations available from all the data we have.

Model choices (and why), consistent with REPORT.md's findings:
- "Made For You"      -> Implicit ALS, n_factors=20 (the validated tuned
                          value from main.py's tune_model_params(), not
                          the original hand-picked 50 -- see REPORT.md S6).
                          The most accurate model in the whole comparison.
- "Friends Are
   Listening To"       -> the friend-based social recommender.
- "Because You
   Listened To X"      -> content-based (raw tag counts; REPORT.md S6 found
                          raw counts beat TF-IDF on accuracy here), seeded
                          on the user's own most-played artist, via
                          similar_items() -- a natural "more like this"
                          framing for a single seed artist.
- "Trending Now"       -> Most Popular.

Every section is wrapped in BackfillRecommender (Most-Popular top-up) so a
thin-data user never sees an empty shelf -- invisibly, unlike the research
prototype's old 📌 markers, because a real product doesn't show its
seams.

Like/skip feedback (feedback_store.py) feeds back into "Made For You" as a
lightweight re-ranking pass -- see _made_for_you_cards -- without
retraining anything: re-fitting implicit ALS per click is far too slow
for an interactive UI, so a "like" instead nudges the existing ALS
ranking towards content-similar artists via the same reciprocal-rank
fusion technique src/social_filtering.py uses for the CF+social hybrid
(REPORT.md S4.5). A "skip" is simpler: the artist is filtered out of
every section, immediately and everywhere, for that listener.
"""

import random
from concurrent.futures import ThreadPoolExecutor

from src import config
from src.data_loading import (
    load_interactions, load_artists, load_tags, load_user_tagged_artists, load_friends,
    build_artist_tag_corpus, get_seen_items,
)
from src.baselines import MostPopularRecommender
from src.content_based import ContentBasedRecommender
from src.matrix_factorization import ImplicitALSRecommender
from src.social_filtering import FriendBasedRecommender
from src.backfill import BackfillRecommender
from . import feedback_store
from .artist_images import get_artist_image_url, get_artist_top_track, get_artist_top_tracks

HOME_SECTION_SIZE = 12
# Kept modest on purpose: each worker thread's default stack is several MB,
# and the free-tier deploy target (512MB total) is shared with pandas/
# scikit-learn/ALS already resident in memory -- a wider pool was observed
# to OOM-kill the single worker process under real traffic.
_IMAGE_POOL = ThreadPoolExecutor(max_workers=4)
# Deliberately separate and smaller than _IMAGE_POOL: firing image and
# track lookups at the same concurrency doubles the burst of requests
# iTunes' (undocumented, fairly aggressive) Search API sees per page load,
# which risks rate-limiting the artwork lookups too. Tracks are cosmetic,
# so they're allowed to trickle in slower instead.
_TRACK_POOL = ThreadPoolExecutor(max_workers=2)


class RecommenderService:
    def __init__(self):
        self.interactions = load_interactions()
        self.artists = load_artists()
        tags = load_tags()
        user_tagged_artists = load_user_tagged_artists()
        self.friends = load_friends()
        tag_corpus = build_artist_tag_corpus(user_tagged_artists, tags)

        self.id_to_name = self.artists.set_index(config.ITEM_COL)[config.NAME_COL].to_dict()
        self.friend_counts = self.friends.groupby(config.USER_COL).size().to_dict()
        top_artist_per_user = (
            self.interactions.sort_values(config.WEIGHT_COL, ascending=False)
            .drop_duplicates(subset=[config.USER_COL], keep="first")
            .set_index(config.USER_COL)[config.ITEM_COL]
        )
        self._top_artist_name = {
            uid: self.id_to_name.get(artist_id) for uid, artist_id in top_artist_per_user.items()
        }

        most_popular = MostPopularRecommender().fit(self.interactions)
        content_based = ContentBasedRecommender(use_tfidf=False).fit(
            self.interactions, self.artists, tag_corpus
        )
        als = ImplicitALSRecommender(n_factors=20, n_iterations=15).fit(self.interactions)
        friend_based = FriendBasedRecommender().fit(self.interactions, self.friends)

        self.most_popular = most_popular
        self.content_based = content_based
        self.made_for_you = BackfillRecommender(als, most_popular)
        self.friends_listening = BackfillRecommender(friend_based, most_popular)

        self.valid_user_ids = sorted(self.interactions[config.USER_COL].unique().tolist())
        self._valid_user_id_set = set(self.valid_user_ids)

    # ---- helpers ----

    def is_valid_user(self, user_id):
        return user_id in self._valid_user_id_set

    def _top_played_artists(self, user_id, n=5):
        rows = self.interactions[self.interactions[config.USER_COL] == user_id]
        rows = rows.sort_values(config.WEIGHT_COL, ascending=False)
        return rows[config.ITEM_COL].head(n).tolist()

    def _artist_card(self, artist_id):
        return {
            "id": int(artist_id),
            "name": self.id_to_name.get(artist_id, f"Artist {artist_id}"),
            "image_url": None,
            "top_track": None,
        }

    def _resolve_images(self, cards):
        """Resolves real artwork and a representative top track for every
        card in one batched, parallel pass. The top track is cosmetic --
        every recommendation here is still artist-level (see module
        docstring / REPORT.md): it just gives each card a real song to
        show instead of reading as a bare artist name."""
        image_futures = [
            _IMAGE_POOL.submit(get_artist_image_url, card["name"]) for card in cards
        ]
        track_futures = [
            _TRACK_POOL.submit(get_artist_top_track, card["name"]) for card in cards
        ]
        for card, future in zip(cards, image_futures):
            try:
                card["image_url"] = future.result(timeout=6)
            except Exception:
                card["image_url"] = None
        for card, future in zip(cards, track_futures):
            try:
                card["top_track"] = future.result(timeout=6)
            except Exception:
                card["top_track"] = None
        return cards

    def _recs_to_cards(self, recs, n, skipped_ids=frozenset()):
        cards = []
        for item_id, _score in recs:
            if item_id in skipped_ids:
                continue
            cards.append({"id": int(item_id), "name": self.id_to_name.get(item_id, f"Artist {item_id}"),
                          "image_url": None, "top_track": None})
            if len(cards) >= n:
                break
        return cards

    # ---- public API ----

    def random_user_id(self):
        return random.choice(self.valid_user_ids)

    def list_users(self, limit=300):
        """A bounded sample for a searchable listener picker -- biased
        toward users with enough history/friends that every home section
        has something real to show (not backfill-only)."""
        n_plays = self.interactions.groupby(config.USER_COL).size()
        scored = sorted(
            self.valid_user_ids,
            key=lambda uid: (self.friend_counts.get(uid, 0) > 0, n_plays.get(uid, 0)),
            reverse=True,
        )
        return [{"id": int(uid), "top_artist": self._top_artist_name.get(uid)}
                for uid in scored[:limit]]

    def record_feedback(self, user_id, artist_id, action):
        feedback_store.record(user_id, artist_id, action)

    def get_artist_tracks(self, artist_id, n=10):
        """Top-n tracks for one artist, for the per-card 'view tracks'
        popup -- a browseable view into an already-recommended artist's
        catalog, not a song-level recommendation (REPORT.md §2/§8)."""
        name = self.id_to_name.get(artist_id, f"Artist {artist_id}")
        return {"artist": {"id": int(artist_id), "name": name}, "tracks": get_artist_top_tracks(name, n=n)}

    def create_playlist(self, user_id, name):
        return feedback_store.create_playlist(user_id, name)

    def delete_playlist(self, user_id, playlist_id):
        feedback_store.delete_playlist(user_id, playlist_id)

    def add_song_to_playlist(self, user_id, playlist_id, artist_id, track):
        return feedback_store.add_song_to_playlist(user_id, playlist_id, {
            "artist_id": int(artist_id),
            "artist_name": self.id_to_name.get(artist_id, f"Artist {artist_id}"),
            "track_id": track["track_id"],
            "track_name": track["track_name"],
            "artwork_url": track.get("artwork_url"),
            "preview_url": track.get("preview_url"),
        })

    def remove_song_from_playlist(self, user_id, playlist_id, track_id):
        feedback_store.remove_song_from_playlist(user_id, playlist_id, track_id)

    def get_playlists(self, user_id):
        return feedback_store.get_playlists(user_id)

    def _made_for_you_cards(self, user_id, n, liked_ids, skipped_ids):
        """ALS recs, re-ranked by a reciprocal-rank-fusion blend with
        content-similarity to whatever the listener has liked so far (see
        module docstring) -- a cheap re-ranking pass, not a retrain."""
        pool = n + len(skipped_ids) + 10
        base_recs = [
            (item_id, score) for item_id, score in
            self.made_for_you.recommend(user_id, self.interactions, n=pool, exclude_seen=True)
            if item_id not in skipped_ids
        ]
        if not liked_ids:
            return self._recs_to_cards(base_recs, n)

        liked_scores = {}
        for liked_id in liked_ids:
            for item_id, sim in self.content_based.similar_items(liked_id, n=30):
                if item_id in skipped_ids:
                    continue
                liked_scores[item_id] = liked_scores.get(item_id, 0.0) + sim
        liked_ranked = sorted(liked_scores.items(), key=lambda kv: -kv[1])

        combined = {}
        for rank, (item_id, _score) in enumerate(base_recs):
            combined[item_id] = combined.get(item_id, 0.0) + 0.7 / (rank + 1)
        for rank, (item_id, _score) in enumerate(liked_ranked):
            combined[item_id] = combined.get(item_id, 0.0) + 0.3 / (rank + 1)

        seen = get_seen_items(self.interactions, user_id)
        ranked = sorted(
            (iid for iid in combined if iid not in seen and iid not in skipped_ids),
            key=lambda iid: -combined[iid],
        )
        return [{"id": int(iid), "name": self.id_to_name.get(iid, f"Artist {iid}"),
                  "image_url": None, "top_track": None}
                for iid in ranked[:n]]

    def get_home_feed(self, user_id, n=HOME_SECTION_SIZE):
        top_artists = self._top_played_artists(user_id, n=6)
        liked_ids, skipped_ids = feedback_store.get(user_id)

        made_for_you = self._made_for_you_cards(user_id, n, liked_ids, skipped_ids)

        friends_listening = []
        if self.friend_counts.get(user_id, 0) > 0:
            pool = n + len(skipped_ids) + 10
            friends_listening = self._recs_to_cards(
                self.friends_listening.recommend(user_id, self.interactions, n=pool, exclude_seen=True),
                n, skipped_ids)

        because_you_listened = None
        for seed_id in top_artists:
            if seed_id in skipped_ids:
                continue
            similar = self.content_based.similar_items(seed_id, n=n + len(skipped_ids) + 10)
            cards = self._recs_to_cards(similar, n, skipped_ids)
            if cards:
                because_you_listened = {
                    "seed_artist": self._artist_card(seed_id),
                    "items": cards,
                }
                break

        pool = n + len(skipped_ids) + 10
        trending = self._recs_to_cards(
            self.most_popular.recommend(user_id, self.interactions, n=pool, exclude_seen=True),
            n, skipped_ids)

        profile = {
            "id": int(user_id),
            "top_artists": [self._artist_card(a) for a in top_artists],
            "n_friends": int(self.friend_counts.get(user_id, 0)),
        }

        all_cards = list(profile["top_artists"]) + made_for_you + friends_listening + trending
        if because_you_listened:
            all_cards.append(because_you_listened["seed_artist"])
            all_cards.extend(because_you_listened["items"])
        self._resolve_images(all_cards)

        return {
            "profile": profile,
            "liked_artist_ids": sorted(int(a) for a in liked_ids),
            "sections": {
                "made_for_you": made_for_you,
                "friends_listening": friends_listening,
                "because_you_listened": because_you_listened,
                "trending": trending,
            },
        }
