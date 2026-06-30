"""Configuration for the Last.fm (HetRec 2011) music recommender prototype.

Dataset: hetrec2011-lastfm-2k
  1,892 users, 17,632 artists, 92,834 user-artist listening-count records,
  11,946 tags, 186,479 user-applied tag assignments, 25,434 directed
  friend-relation pairs (12,717 bidirectional friendships).
Source: GroupLens / 2nd Workshop on Information Heterogeneity and Fusion in
Recommender Systems (HetRec 2011), released for non-commercial research use.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
EXAMPLES_DIR = RESULTS_DIR / "examples"

# Raw HetRec2011 Last.fm files (tab-separated).
INTERACTIONS_PATH = RAW_DATA_DIR / "user_artists.dat"          # userID, artistID, weight (play count)
ARTISTS_PATH = RAW_DATA_DIR / "artists.dat"                     # id, name, url, pictureURL
TAGS_PATH = RAW_DATA_DIR / "tags.dat"                            # tagID, tagValue
USER_TAGGED_ARTISTS_PATH = RAW_DATA_DIR / "user_taggedartists.dat"  # userID, artistID, tagID, day, month, year
USER_FRIENDS_PATH = RAW_DATA_DIR / "user_friends.dat"            # userID, friendID

# Column names used throughout the project (kept MovieLens-style generic
# names so the overall pipeline shape -- load -> split -> fit -> evaluate
# -- looks the same regardless of domain).
USER_COL = "userID"
ITEM_COL = "artistID"
WEIGHT_COL = "weight"        # raw play count -- this is implicit feedback, NOT a 1-5 star rating
NAME_COL = "name"

TOP_K = 10
RANDOM_STATE = 42

# Implicit feedback has no rating scale, so there is no fixed "relevance
# threshold" like MovieLens' "rated >= 4 stars". Instead, a held-out
# interaction counts as relevant if its play count is at or above the
# user's own median play count in the *training* set -- i.e. "an artist
# this user actually favoured", not a one-off background play picked up
# by a shuffle-play session. See data_loading.get_relevant_items.
RELEVANCE_PERCENTILE = 0.5

# Users/artists below these thresholds are kept in training (so they still
# contribute signal to similarity/factor estimation) but are excluded from
# the evaluable test split / from CF neighborhoods, since there isn't
# enough data to support a reliable similarity estimate or a meaningful
# held-out test slice for them.
MIN_INTERACTIONS_PER_USER = 5
MIN_LISTENERS_PER_ARTIST = 5

# Confidence-weighting hyperparameter for implicit-feedback ALS:
# confidence(u,i) = 1 + ALPHA * log1p(weight(u,i))
IMPLICIT_ALPHA = 2.0
