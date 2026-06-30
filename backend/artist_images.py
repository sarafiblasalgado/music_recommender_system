"""Real artist artwork lookup (iTunes Search) and per-artist track lookup
(Deezer) -- both keyless, no account/API-key setup required.

The HetRec2011 dataset ships a `pictureURL` column in artists.dat, but
those are 2011-era Last.fm CDN links (userserve-ak.last.fm) that no
longer resolve -- the DNS itself is dead. iTunes Search returns a 100x100
artwork thumbnail per match, which we upscale to 600x600 by swapping the
size segment in the URL (a well-known iTunes CDN convention -- the file
exists at multiple sizes under the same path).

Track lookups (top track / top-10 tracks, used for the card subtitle and
the per-artist "view tracks" popup) use Deezer's public API instead of
iTunes Search: Deezer has an actual artist-top-tracks endpoint (ranked by
real popularity, not iTunes' relevance-ranked search match for an
`entity=song` query), needs no API key, and -- the reason for the
switch -- iTunes' rate limit on `entity=song` queries turned out to be
far more aggressive and flaky than on its plain artist/image search,
enough to make the track features unreliable. Deezer's public endpoints
have a documented, generous limit (~50 req/5s/IP) that this app's usage
never gets close to.

Lookups are cached to disk (data/processed/*.json) so a restart doesn't
re-hit the network for every artist. A *confirmed* miss (API responded,
no match in the results) is cached so we don't retry a dead match every
time -- but a *failed request* (network error, timeout, rate limit) is
deliberately NOT cached, since that's a transient condition, not a fact
about the artist; caching it would permanently blank out an artist that
would have resolved fine a minute later. The frontend falls back to a
generated gradient avatar/card for anything with no cached/resolvable
data, so a slow or unavailable network never blocks the UI.
"""

import json
import threading
from pathlib import Path

import requests

CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "artist_image_cache.json"
TRACKLIST_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "artist_tracklist_cache.json"
DEEZER_ID_CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "deezer_artist_id_cache.json"

_lock = threading.Lock()
_cache = {}
_tracklist_lock = threading.Lock()
_tracklist_cache = {}
_deezer_id_lock = threading.Lock()
_deezer_id_cache = {}


def _load_json_cache(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_json_cache(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _load_cache():
    global _cache
    _cache = _load_json_cache(CACHE_PATH)


def _save_cache():
    _save_json_cache(CACHE_PATH, _cache)


def _load_tracklist_cache():
    global _tracklist_cache
    _tracklist_cache = _load_json_cache(TRACKLIST_CACHE_PATH)


def _save_tracklist_cache():
    _save_json_cache(TRACKLIST_CACHE_PATH, _tracklist_cache)


def _load_deezer_id_cache():
    global _deezer_id_cache
    _deezer_id_cache = _load_json_cache(DEEZER_ID_CACHE_PATH)


def _save_deezer_id_cache():
    _save_json_cache(DEEZER_ID_CACHE_PATH, _deezer_id_cache)


_load_cache()
_load_tracklist_cache()
_load_deezer_id_cache()


def get_artist_image_url(artist_name, timeout=4):
    """Return a 600x600 artwork URL for the artist, or None if no match /
    the lookup failed. A confirmed no-match is cached across calls and
    process restarts; a failed request is not (see module docstring)."""
    if artist_name in _cache:
        return _cache[artist_name]

    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": artist_name, "media": "music", "limit": 1},
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except (requests.RequestException, ValueError):
        return None

    image_url = None
    if results:
        thumb = results[0].get("artworkUrl100")
        if thumb:
            image_url = thumb.replace("100x100bb", "600x600bb")

    with _lock:
        _cache[artist_name] = image_url
        _save_cache()
    return image_url


def _resolve_deezer_artist_id(artist_name, timeout=4):
    """Resolve a Deezer artist id for a name, or None. Unlike the other
    lookups in this module, a miss here is deliberately NEVER cached, only
    a confirmed match is: this step gates everything downstream (no id ->
    no tracks at all for that artist), so caching a false negative is far
    more costly than re-querying a genuinely-obscure artist every time.
    An empty `data: []` on a well-known artist has been observed here from
    a transient/load condition, not just a real "Deezer has never heard of
    this artist" -- treating that as permanent would be wrong."""
    if artist_name in _deezer_id_cache:
        return _deezer_id_cache[artist_name]

    try:
        resp = requests.get(
            "https://api.deezer.com/search/artist",
            params={"q": artist_name, "limit": 1},
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("data", [])
    except (requests.RequestException, ValueError):
        return None

    if not results:
        return None

    artist_id = results[0]["id"]
    with _deezer_id_lock:
        _deezer_id_cache[artist_name] = artist_id
        _save_deezer_id_cache()
    return artist_id


def get_artist_top_tracks(artist_name, n=10, timeout=6):
    """Return up to n {track_id, name, artwork_url} dicts for the artist,
    ranked by Deezer's actual artist-top-tracks endpoint (real popularity,
    not a search-relevance guess). Powers both the card subtitle
    (n=1, via get_artist_top_track) and the per-artist "view tracks"
    popup a listener opens by clicking a card (ArtistTracksModal). Still
    doesn't change what's recommended (REPORT.md §2/§8) -- it's a
    browseable view into one already-recommended artist's catalog, not a
    different granularity of recommendation."""
    cache_key = f"{artist_name}::{n}"
    if cache_key in _tracklist_cache:
        return _tracklist_cache[cache_key]

    # Resolving the artist id never caches a miss (see
    # _resolve_deezer_artist_id), so artist_id is None here either because
    # this is a genuine no-match or a transient failure -- either way,
    # don't cache this top-tracks lookup off the back of it.
    artist_id = _resolve_deezer_artist_id(artist_name, timeout=timeout)
    if artist_id is None:
        return []

    try:
        resp = requests.get(
            f"https://api.deezer.com/artist/{artist_id}/top",
            params={"limit": n},
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json().get("data", [])
    except (requests.RequestException, ValueError):
        return []

    tracks = []
    for r in results:
        track_id, track_name = r.get("id"), r.get("title")
        if track_id is None or track_name is None:
            continue
        tracks.append({
            "track_id": track_id,
            "name": track_name,
            "artwork_url": (r.get("album") or {}).get("cover_medium"),
            "preview_url": r.get("preview"),
        })

    # A resolved artist id came from a real Deezer artist match, so an
    # empty top-tracks result is very unlikely to be genuine -- same
    # "don't permanently cache a probably-transient miss" reasoning as
    # the artist-id step above. Only cache once there's something to cache.
    if tracks:
        with _tracklist_lock:
            _tracklist_cache[cache_key] = tracks
            _save_tracklist_cache()
    return tracks


def get_artist_top_track(artist_name, timeout=4):
    """Return a representative track title for the artist, or None. Purely
    cosmetic, surfaced as a card subtitle so the UI doesn't read as a flat
    list of artist names -- it does NOT change what's actually being
    recommended (see get_artist_top_tracks docstring). Delegates to
    get_artist_top_tracks(n=1) so both share one cache and one code path."""
    tracks = get_artist_top_tracks(artist_name, n=1, timeout=timeout)
    return tracks[0]["name"] if tracks else None
