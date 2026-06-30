"""Data loading and preprocessing for the Last.fm HetRec2011 dataset.

This is implicit-feedback data: there is no 1-5 star rating, only a play
count per (user, artist) pair. Every preprocessing choice here exists to
turn that single noisy count into something the rest of the pipeline can
use, and to fold in the dataset's two extra "layers" beyond plain
listening history -- user-supplied tags and the social friend graph.

Design choices:

- **Log-transform play counts.** Raw counts are extremely heavy-tailed
  (median 260 plays, max 352,698 in this dataset) -- see EDA. A user who
  played one artist 300,000 times isn't "1000x more interested" in it than
  an artist they played 300 times; log1p compresses this onto a usable
  scale and is the standard treatment for implicit play-count data.
- **Per-user train/test split** (same rationale as an explicit-ratings
  project): guarantees every evaluable user has both training history and
  held-out ground truth, instead of risking a user landing entirely on one
  side of a single global random split.
- **Relevance without a rating scale.** "Relevant" for ranking metrics is
  defined relative to each user's own training behaviour: a held-out
  interaction is relevant if its play count is at or above that user's
  median training play count, i.e. "one of the artists this user actually
  favoured" rather than a one-off track that shuffled into a session.
"""

import numpy as np
import pandas as pd

from . import config


def load_interactions(path=config.INTERACTIONS_PATH):
    """Load user-artist listening counts (implicit feedback).

    Returns a DataFrame with columns: userID, artistID, weight, log_weight
    """
    interactions = pd.read_csv(str(path), sep="\t")
    required = {config.USER_COL, config.ITEM_COL, config.WEIGHT_COL}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"interactions file is missing required columns: {missing}")

    interactions[config.USER_COL] = interactions[config.USER_COL].astype(int)
    interactions[config.ITEM_COL] = interactions[config.ITEM_COL].astype(int)
    interactions[config.WEIGHT_COL] = interactions[config.WEIGHT_COL].astype(float)
    interactions["log_weight"] = np.log1p(interactions[config.WEIGHT_COL])
    return interactions.reset_index(drop=True)


def load_artists(path=config.ARTISTS_PATH):
    """Load artist metadata. Returns columns: artistID, name, url, pictureURL.

    Encoded as UTF-8 in this dataset (unlike tags.dat/user_taggedartists.dat,
    which are latin-1) -- reading it as latin-1 silently "succeeds" but
    mangles every non-ASCII artist name into mojibake, e.g. "Röyksopp" ->
    "RÃ¶yksopp". Verified by checking which encoding round-trips cleanly.
    """
    artists = pd.read_csv(str(path), sep="\t", encoding="utf-8")
    artists = artists.rename(columns={"id": config.ITEM_COL})
    artists[config.ITEM_COL] = artists[config.ITEM_COL].astype(int)
    # A handful of duplicate artist names exist in this dataset under
    # different IDs (mis-tagged scrobbles); we keep them as distinct items
    # since they have distinct IDs and distinct listening histories.
    return artists.reset_index(drop=True)


def load_tags(path=config.TAGS_PATH):
    """Load the tag vocabulary. Returns columns: tagID, tagValue."""
    tags = pd.read_csv(str(path), sep="\t", encoding="latin-1")
    return tags.reset_index(drop=True)


def load_user_tagged_artists(path=config.USER_TAGGED_ARTISTS_PATH):
    """Load (user, artist, tag) assignments used to build content features."""
    uta = pd.read_csv(str(path), sep="\t", encoding="latin-1")
    uta[config.ITEM_COL] = uta[config.ITEM_COL].astype(int)
    return uta.reset_index(drop=True)


def load_friends(path=config.USER_FRIENDS_PATH):
    """Load the (directed) social friend graph. Returns columns: userID, friendID."""
    friends = pd.read_csv(str(path), sep="\t", encoding="latin-1")
    friends[config.USER_COL] = friends[config.USER_COL].astype(int)
    friends["friendID"] = friends["friendID"].astype(int)
    return friends.reset_index(drop=True)


def build_artist_tag_corpus(user_tagged_artists, tags):
    """Aggregate every tag ever applied to each artist into a bag-of-tags
    string, e.g. artist 52 -> "rock alternative_rock 90s ...". Used as the
    text corpus for the content-based TF-IDF vectorizer.
    """
    tag_lookup = tags.set_index("tagID")["tagValue"].to_dict()

    def to_token(tag_id):
        value = str(tag_lookup.get(tag_id, "")).strip().lower()
        return value.replace(" ", "_") if value else ""

    uta = user_tagged_artists.copy()
    uta["token"] = uta["tagID"].map(to_token)
    uta = uta[uta["token"] != ""]
    corpus = uta.groupby(config.ITEM_COL)["token"].apply(lambda s: " ".join(s))
    return corpus  # Series indexed by artistID -> tag text


def describe_dataset(interactions, artists=None, friends=None, verbose=True):
    """Compute (and optionally print) dataset statistics across all three
    data layers: listening behaviour, and the social friend graph."""
    n_users = interactions[config.USER_COL].nunique()
    n_artists = interactions[config.ITEM_COL].nunique()
    n_interactions = len(interactions)
    sparsity = 1 - n_interactions / (n_users * n_artists)

    plays_per_user = interactions.groupby(config.USER_COL)[config.WEIGHT_COL].sum()
    interactions_per_user = interactions.groupby(config.USER_COL).size()
    listeners_per_artist = interactions.groupby(config.ITEM_COL).size()
    plays_per_artist = interactions.groupby(config.ITEM_COL)[config.WEIGHT_COL].sum()

    most_active_users = (
        plays_per_user.sort_values(ascending=False).head(10)
        .rename("total_plays").rename_axis(config.USER_COL).reset_index()
    )
    most_popular_artists = (
        listeners_per_artist.sort_values(ascending=False).head(10)
        .rename("n_listeners").rename_axis(config.ITEM_COL).reset_index()
    )
    if artists is not None:
        most_popular_artists = most_popular_artists.merge(
            artists[[config.ITEM_COL, config.NAME_COL]], on=config.ITEM_COL, how="left"
        )

    stats = {
        "n_users": n_users,
        "n_artists": n_artists,
        "n_interactions": n_interactions,
        "sparsity": sparsity,
        "density_pct": (1 - sparsity) * 100,
        "mean_weight": interactions[config.WEIGHT_COL].mean(),
        "median_weight": interactions[config.WEIGHT_COL].median(),
        "max_weight": interactions[config.WEIGHT_COL].max(),
        "avg_interactions_per_user": interactions_per_user.mean(),
        "median_interactions_per_user": interactions_per_user.median(),
        "avg_listeners_per_artist": listeners_per_artist.mean(),
        "median_listeners_per_artist": listeners_per_artist.median(),
        "most_active_users": most_active_users,
        "most_popular_artists": most_popular_artists,
    }

    if friends is not None:
        n_friend_users = friends[config.USER_COL].nunique()
        degree = friends.groupby(config.USER_COL).size()
        stats["n_friend_edges_directed"] = len(friends)
        stats["n_users_with_friends"] = n_friend_users
        stats["avg_friends_per_user"] = degree.mean()
        stats["median_friends_per_user"] = degree.median()

    if verbose:
        print("=" * 60)
        print("DATASET SUMMARY -- Last.fm HetRec2011")
        print("=" * 60)
        print(f"Users:                     {stats['n_users']:,}")
        print(f"Artists:                   {stats['n_artists']:,}")
        print(f"Listening interactions:    {stats['n_interactions']:,}")
        print(f"Sparsity:                  {stats['sparsity']:.4%}  (density {stats['density_pct']:.4f}%)")
        print(f"Play count (mean/median/max): {stats['mean_weight']:.1f} / "
              f"{stats['median_weight']:.0f} / {stats['max_weight']:.0f}")
        print(f"Interactions/user (mean/median): {stats['avg_interactions_per_user']:.1f} / "
              f"{stats['median_interactions_per_user']:.0f}")
        print(f"Listeners/artist (mean/median):  {stats['avg_listeners_per_artist']:.1f} / "
              f"{stats['median_listeners_per_artist']:.0f}")
        if friends is not None:
            print("-" * 60)
            print(f"Social layer: {stats['n_users_with_friends']:,} users have >=1 friend, "
                  f"{stats['n_friend_edges_directed']:,} directed edges, "
                  f"avg {stats['avg_friends_per_user']:.1f} friends/user")
        print("-" * 60)
        print("Top 5 most active users (by total plays):")
        print(most_active_users.head(5).to_string(index=False))
        print("-" * 60)
        print("Top 5 most popular artists (by # listeners):")
        cols = [config.ITEM_COL, "n_listeners"] + ([config.NAME_COL] if artists is not None else [])
        print(most_popular_artists[cols].head(5).to_string(index=False))
        print("=" * 60)

    return stats


def train_test_split_interactions(interactions, test_size=0.2, random_state=config.RANDOM_STATE,
                                   min_interactions_per_user=config.MIN_INTERACTIONS_PER_USER):
    """Per-user random split of listening interactions (same rationale as
    an explicit-ratings split -- see module docstring)."""
    rng = np.random.RandomState(random_state)
    train_parts, test_parts = [], []

    for user_id, group in interactions.groupby(config.USER_COL):
        idx = group.index.to_numpy().copy()
        if len(idx) < min_interactions_per_user:
            train_parts.append(idx)
            continue
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        n_test = min(n_test, len(idx) - 1)
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

    train = interactions.loc[train_idx].sort_values([config.USER_COL, config.ITEM_COL]).reset_index(drop=True)
    test = interactions.loc[test_idx].sort_values([config.USER_COL, config.ITEM_COL]).reset_index(drop=True)
    return train, test


def get_interaction_timestamps(user_tagged_artists):
    """Approximate a (user, artist) listening interaction's timestamp from
    the *earliest* tag-application event for that pair, as (year*12 +
    month) -- a sortable, monotonic proxy.

    `user_artists.dat` (the play-count data every model in this project
    trains on) carries no timestamp at all -- only `user_taggedartists.dat`
    does, and only for the (user, artist) pairs that user happened to tag.
    This is therefore a coverage-limited proxy, not a ground-truth
    listening date: see `train_test_split_interactions_temporal` and
    REPORT.md S10 for how that limitation is handled rather than ignored.
    The `day` field is dropped -- it's 1 for the overwhelming majority of
    rows in this dataset (effectively unset), so month/year is the
    reliable resolution.
    """
    df = user_tagged_artists.copy()
    df["yyyymm"] = df["year"] * 12 + df["month"]
    earliest = df.groupby([config.USER_COL, config.ITEM_COL])["yyyymm"].min()
    return earliest  # Series indexed by (userID, artistID) -> sortable timestamp proxy


def train_test_split_interactions_temporal(interactions, timestamps, test_size=0.2,
                                            min_interactions_per_user=config.MIN_INTERACTIONS_PER_USER):
    """Per-user temporal split: each user's *earliest* interactions (by the
    tag-timestamp proxy) go to train, their *most recent* go to test --
    the standard "predict the future from the past" protocol, applied only
    to the subset of (user, artist) pairs where a timestamp proxy exists
    at all (see get_interaction_timestamps). Restricting to that subset
    -- rather than silently mixing timestamped and un-timestamped rows --
    is what makes a fair comparison against a random split on the *same*
    data possible (temporal_split_check.py does exactly that comparison).
    """
    df = interactions.copy()
    df["yyyymm"] = list(
        zip(df[config.USER_COL], df[config.ITEM_COL])
    )
    df["yyyymm"] = df["yyyymm"].map(timestamps.to_dict())
    df = df.dropna(subset=["yyyymm"])

    train_parts, test_parts = [], []
    for user_id, group in df.groupby(config.USER_COL):
        group = group.sort_values("yyyymm", kind="stable")
        idx = group.index.to_numpy()
        if len(idx) < min_interactions_per_user:
            train_parts.append(idx)
            continue
        n_test = max(1, int(round(len(idx) * test_size)))
        n_test = min(n_test, len(idx) - 1)
        train_parts.append(idx[:-n_test])
        test_parts.append(idx[-n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

    train = interactions.loc[train_idx].sort_values([config.USER_COL, config.ITEM_COL]).reset_index(drop=True)
    test = interactions.loc[test_idx].sort_values([config.USER_COL, config.ITEM_COL]).reset_index(drop=True)
    return train, test


def get_seen_items(interactions, user_id):
    """Return the set of artist IDs the user has already listened to."""
    return set(interactions.loc[interactions[config.USER_COL] == user_id, config.ITEM_COL])


def get_user_log_weight_means(interactions):
    """Mean log-play-count per user, used to center content-based profiles."""
    return interactions.groupby(config.USER_COL)["log_weight"].mean()


def get_user_median_weights(interactions):
    """Median raw play count per user, used as the per-user relevance bar."""
    return interactions.groupby(config.USER_COL)[config.WEIGHT_COL].median()


def get_user_quantile_weights(interactions, q=0.5):
    """Generalizes get_user_median_weights to an arbitrary quantile.
    q=0.5 reproduces the project's default relevance bar; other values let
    relevance_threshold_check.py test whether the headline results in
    REPORT.md S6 are sensitive to that specific (otherwise unexamined)
    choice of threshold."""
    return interactions.groupby(config.USER_COL)[config.WEIGHT_COL].quantile(q)


def get_relevant_items(test_interactions, user_id, user_median_weights, global_median_weight):
    """A held-out interaction counts as 'relevant' if its play count is at
    or above the user's own median training play count (falls back to the
    global median for users with no training median available)."""
    threshold = user_median_weights.get(user_id, global_median_weight)
    user_test = test_interactions[
        (test_interactions[config.USER_COL] == user_id)
        & (test_interactions[config.WEIGHT_COL] >= threshold)
    ]
    return set(user_test[config.ITEM_COL])
