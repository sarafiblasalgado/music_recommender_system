// In production this is set via the VITE_API_BASE environment variable
// (e.g. Vercel project settings) pointing at the deployed backend; falls
// back to localhost for local dev with no setup required.
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function getJSON(path) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${path} -> ${res.status}`);
  }
  return res.json();
}

export function listListeners(limit = 300) {
  return getJSON(`/api/users?limit=${limit}`);
}

export function randomListenerId() {
  return getJSON("/api/users/random").then((data) => data.id);
}

export function getHomeFeed(userId) {
  return getJSON(`/api/home/${userId}`);
}

export function sendFeedback(userId, artistId, action) {
  return fetch(`${API_BASE}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, artist_id: artistId, action }),
  });
}

export function getArtistTracks(artistId) {
  return getJSON(`/api/artist/${artistId}/tracks`);
}

export function getPlaylists(userId) {
  return getJSON(`/api/playlists/${userId}`).then((data) => data.playlists);
}

export function createPlaylist(userId, name) {
  return fetch(`${API_BASE}/api/playlists`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, name }),
  }).then((res) => res.json());
}

export function deletePlaylist(userId, playlistId) {
  return fetch(`${API_BASE}/api/playlists/${userId}/${playlistId}`, { method: "DELETE" });
}

export function addSongToPlaylist(userId, playlistId, artistId, track) {
  return fetch(`${API_BASE}/api/playlists/${playlistId}/songs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      artist_id: artistId,
      track_id: track.track_id,
      track_name: track.name,
      artwork_url: track.artwork_url,
      preview_url: track.preview_url,
    }),
  });
}

export function removeSongFromPlaylist(userId, playlistId, trackId) {
  return fetch(`${API_BASE}/api/playlists/${playlistId}/songs/remove`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, track_id: trackId }),
  });
}
