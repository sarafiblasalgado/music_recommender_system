"""Per-listener like/skip feedback for the prototype's re-ranking loop,
plus named, multi-playlist song collections.

Artist-level like/skip is a product feature, not a modelling one: it does
not retrain or mutate any of the fitted recommenders (re-fitting implicit
ALS per click is far too slow for an interactive UI). Instead
RecommenderService reads this store on every request and folds it into
"Made For You" as a lightweight re-ranking pass -- see
recommender_service._made_for_you_cards.

Song-level playlists are simpler and deliberately don't feed back into any
recommender: clicking an artist card opens its top-10 tracks (see
artist_images.get_artist_top_tracks), and adding a track to a playlist
there just builds the listener's own collection (REPORT.md §8/§10) -- a
way to act at song granularity on top of artist-granularity
recommendations, without pretending the underlying ranking is song-level.
A listener can have any number of named playlists, not just one.

Persisted to disk (data/processed/user_feedback.json) so feedback survives
a backend restart, the same pattern as artist_images.py's image cache.
Process-memory + a JSON file is deliberately the entire "infrastructure"
here -- this is a single-process prototype, not a production service.
"""

import json
import threading
import uuid
from pathlib import Path

STORE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "user_feedback.json"
_lock = threading.Lock()
_store = {}


def _load():
    global _store
    if STORE_PATH.exists():
        try:
            raw = json.loads(STORE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            raw = {}
    else:
        raw = {}

    _store = {}
    for uid, v in raw.items():
        playlists = v.get("playlists")
        if playlists is None:
            # Pre-multi-playlist data had one flat `liked_songs` collection
            # per user -- migrate it into a single named playlist instead
            # of dropping it.
            old_songs = v.get("liked_songs", {})
            playlists = {}
            if old_songs:
                playlists[uuid.uuid4().hex[:12]] = {"name": "My Playlist", "songs": dict(old_songs)}
        _store[int(uid)] = {
            "liked": set(v.get("liked", [])),
            "skipped": set(v.get("skipped", [])),
            "playlists": playlists,
        }


def _save():
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(uid): {
            "liked": sorted(v["liked"]),
            "skipped": sorted(v["skipped"]),
            "playlists": v["playlists"],
        }
        for uid, v in _store.items()
    }
    STORE_PATH.write_text(json.dumps(serializable))


_load()


def _entry(user_id):
    return _store.setdefault(user_id, {"liked": set(), "skipped": set(), "playlists": {}})


def record(user_id, artist_id, action):
    """action: 'like' | 'skip' | 'clear' (clear removes any like/skip)."""
    with _lock:
        entry = _entry(user_id)
        if action == "like":
            entry["liked"].add(artist_id)
            entry["skipped"].discard(artist_id)
        elif action == "skip":
            entry["skipped"].add(artist_id)
            entry["liked"].discard(artist_id)
        elif action == "clear":
            entry["liked"].discard(artist_id)
            entry["skipped"].discard(artist_id)
        else:
            raise ValueError(f"unknown feedback action: {action}")
        _save()


def get(user_id):
    """Return (liked_set, skipped_set) for a user -- empty sets if none yet."""
    entry = _store.get(user_id, {"liked": set(), "skipped": set()})
    return entry["liked"], entry["skipped"]


def create_playlist(user_id, name):
    """Create a new, empty named playlist for this listener. Returns its id."""
    with _lock:
        entry = _entry(user_id)
        playlist_id = uuid.uuid4().hex[:12]
        entry["playlists"][playlist_id] = {"name": name, "songs": {}}
        _save()
    return playlist_id


def delete_playlist(user_id, playlist_id):
    with _lock:
        entry = _entry(user_id)
        entry["playlists"].pop(playlist_id, None)
        _save()


def add_song_to_playlist(user_id, playlist_id, song):
    """song: {artist_id, artist_name, track_id, track_name, artwork_url,
    preview_url}. No-op if the playlist doesn't exist (e.g. deleted in
    another tab) -- returns whether it succeeded."""
    with _lock:
        entry = _entry(user_id)
        playlist = entry["playlists"].get(playlist_id)
        if playlist is None:
            return False
        playlist["songs"][str(song["track_id"])] = song
        _save()
    return True


def remove_song_from_playlist(user_id, playlist_id, track_id):
    with _lock:
        entry = _entry(user_id)
        playlist = entry["playlists"].get(playlist_id)
        if playlist is None:
            return False
        playlist["songs"].pop(str(track_id), None)
        _save()
    return True


def get_playlists(user_id):
    """Return this listener's playlists as [{id, name, songs: [...]}, ...]."""
    entry = _store.get(user_id, {"playlists": {}})
    return [
        {"id": pid, "name": p["name"], "songs": list(p["songs"].values())}
        for pid, p in entry.get("playlists", {}).items()
    ]
