"""API for the consumer-facing recommender prototype (React frontend).

Run with:  uvicorn backend.main:app --reload --port 8000

Loads the dataset and fits every model once at startup (a few seconds for
everything except implicit ALS, ~10-20s), then serves a small JSON API:
no recommendation logic, model names, or evaluation metrics are exposed
here beyond what's needed to render a listener's home feed -- that
analysis lives in REPORT.md, not the product.
"""

import os
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .recommender_service import RecommenderService

app = FastAPI(title="Last.fm Recommender API")

# ALLOWED_ORIGINS (comma-separated) lets the deployed frontend's exact
# origin be configured without a code change; unset locally, where the
# two hardcoded dev origins are enough. No cookies/auth are used by this
# API, so a public deploy with no ALLOWED_ORIGINS set falls back to "*"
# rather than failing closed -- there's nothing sensitive to protect.
_origins_env = os.environ.get("ALLOWED_ORIGINS")
_allow_origins = _origins_env.split(",") if _origins_env else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

service = RecommenderService()


class FeedbackRequest(BaseModel):
    user_id: int
    artist_id: int
    action: Literal["like", "skip", "clear"]


class CreatePlaylistRequest(BaseModel):
    user_id: int
    name: str


class AddSongRequest(BaseModel):
    user_id: int
    artist_id: int
    track_id: int
    track_name: str
    artwork_url: Optional[str] = None
    preview_url: Optional[str] = None


class RemoveSongRequest(BaseModel):
    user_id: int
    track_id: int


@app.get("/api/users")
def list_users(limit: int = 300):
    return service.list_users(limit=limit)


@app.get("/api/users/random")
def random_user():
    return {"id": service.random_user_id()}


@app.get("/api/home/{user_id}")
def home_feed(user_id: int):
    if not service.is_valid_user(user_id):
        raise HTTPException(status_code=404, detail=f"No listener with id {user_id}")
    return service.get_home_feed(user_id)


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    if not service.is_valid_user(req.user_id):
        raise HTTPException(status_code=404, detail=f"No listener with id {req.user_id}")
    service.record_feedback(req.user_id, req.artist_id, req.action)
    return {"status": "ok"}


@app.get("/api/artist/{artist_id}/tracks")
def artist_tracks(artist_id: int):
    return service.get_artist_tracks(artist_id)


@app.get("/api/playlists/{user_id}")
def list_playlists(user_id: int):
    if not service.is_valid_user(user_id):
        raise HTTPException(status_code=404, detail=f"No listener with id {user_id}")
    return {"playlists": service.get_playlists(user_id)}


@app.post("/api/playlists")
def create_playlist(req: CreatePlaylistRequest):
    if not service.is_valid_user(req.user_id):
        raise HTTPException(status_code=404, detail=f"No listener with id {req.user_id}")
    name = req.name.strip() or "Untitled playlist"
    playlist_id = service.create_playlist(req.user_id, name)
    return {"id": playlist_id, "name": name}


@app.delete("/api/playlists/{user_id}/{playlist_id}")
def delete_playlist(user_id: int, playlist_id: str):
    if not service.is_valid_user(user_id):
        raise HTTPException(status_code=404, detail=f"No listener with id {user_id}")
    service.delete_playlist(user_id, playlist_id)
    return {"status": "ok"}


@app.post("/api/playlists/{playlist_id}/songs")
def add_song_to_playlist(playlist_id: str, req: AddSongRequest):
    if not service.is_valid_user(req.user_id):
        raise HTTPException(status_code=404, detail=f"No listener with id {req.user_id}")
    ok = service.add_song_to_playlist(req.user_id, playlist_id, req.artist_id, {
        "track_id": req.track_id, "track_name": req.track_name,
        "artwork_url": req.artwork_url, "preview_url": req.preview_url,
    })
    if not ok:
        raise HTTPException(status_code=404, detail=f"No playlist with id {playlist_id}")
    return {"status": "ok"}


@app.post("/api/playlists/{playlist_id}/songs/remove")
def remove_song_from_playlist(playlist_id: str, req: RemoveSongRequest):
    if not service.is_valid_user(req.user_id):
        raise HTTPException(status_code=404, detail=f"No listener with id {req.user_id}")
    service.remove_song_from_playlist(req.user_id, playlist_id, req.track_id)
    return {"status": "ok"}
