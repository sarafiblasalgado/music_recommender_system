import { useEffect, useState } from "react";
import Modal from "./Modal";
import AddToPlaylistMenu from "./AddToPlaylistMenu";
import { getArtistTracks } from "../api";
import { gradientFor } from "../avatarColor";

function HeartIcon({ filled }) {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2">
      <path d="M12 21s-7.5-4.6-10-9.1C0.3 8.2 1.8 4.5 5.3 3.6c2-.5 4 .3 5.2 2 .9-1.7 2.9-2.5 4.9-2 3.5.9 5 4.6 3.3 8.3C19.5 16.4 12 21 12 21z" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4 ml-0.5" fill="currentColor">
      <path d="M8 5v14l11-7z" />
    </svg>
  );
}

function PauseIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor">
      <path d="M7 5h4v14H7zM13 5h4v14h-4z" />
    </svg>
  );
}

// Opened by clicking an artist card -- a browseable list of that artist's
// top tracks (Deezer, same lookup used for the card subtitle), so a
// listener can act at song granularity on top of an artist-level
// recommendation. The "add" button opens a small menu of the listener's
// playlists (a track can be in several, or none) -- see
// AddToPlaylistMenu. Clicking the artwork plays a 30s preview via the
// shared MiniPlayer. Neither feeds back into any recommender -- see
// REPORT.md S2/S8/S10.
export default function ArtistTracksModal({
  artist, playlists, onTogglePlaylist, onCreatePlaylistAndAdd, onClose,
  currentTrackId, isPlaying, onPlayToggle,
}) {
  const [tracks, setTracks] = useState(null);
  const [error, setError] = useState(false);
  const [retryToken, setRetryToken] = useState(0);
  const [openMenuTrackId, setOpenMenuTrackId] = useState(null);

  useEffect(() => {
    if (!artist) return;
    setTracks(null);
    setError(false);
    setOpenMenuTrackId(null);
    getArtistTracks(artist.id)
      .then((data) => setTracks(data.tracks || []))
      .catch(() => setError(true));
  }, [artist, retryToken]);

  if (!artist) return null;

  // iTunes Search's rate limit is undocumented and, in practice, flaky
  // rather than a hard cutoff -- the same query can 200 a minute after it
  // 403'd. An empty/failed result here is shown as possibly-transient
  // (not "this artist has no tracks") with a one-click retry, rather than
  // silently looking like a dead end.
  function retry() {
    setRetryToken((n) => n + 1);
  }

  return (
    <Modal title={`Top tracks · ${artist.name}`} onClose={onClose}>
      {tracks === null && !error && (
        <div className="px-3 py-8 text-center text-sm text-zinc-500">Loading tracks…</div>
      )}
      {error && (
        <div className="px-3 py-8 text-center text-sm text-zinc-500">
          <p>Couldn't load tracks for {artist.name} right now.</p>
          <button
            type="button"
            onClick={retry}
            className="mt-3 text-violet-300 hover:text-violet-200 underline underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 rounded"
          >
            Try again
          </button>
        </div>
      )}
      {tracks && tracks.length === 0 && (
        <div className="px-3 py-8 text-center text-sm text-zinc-500">
          <p>No tracks found for {artist.name} right now.</p>
          <button
            type="button"
            onClick={retry}
            className="mt-3 text-violet-300 hover:text-violet-200 underline underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 rounded"
          >
            Try again
          </button>
        </div>
      )}
      {tracks && tracks.length > 0 && (
        <ul className="divide-y divide-white/5">
          {tracks.map((track) => {
            const inAnyPlaylist = playlists.some((p) => p.songs.some((s) => s.track_id === track.track_id));
            const playing = currentTrackId === track.track_id && isPlaying;
            return (
              // Deliberately not using the .animate-fade-up entrance
              // animation here, unlike other lists: a CSS animation that
              // ends with a "held" transform (even an identity one, via
              // fill-mode: both) creates a stacking context, which broke
              // the AddToPlaylistMenu popover's z-index against later
              // sibling rows. The modal's own pop-in animation already
              // makes this list feel alive on open.
              <li
                key={track.track_id}
                className="relative flex items-center gap-3 px-3 py-2.5"
              >
                <button
                  type="button"
                  onClick={() => track.preview_url && onPlayToggle({ ...track, artist_name: artist.name })}
                  disabled={!track.preview_url}
                  aria-label={playing ? `Pause ${track.name}` : `Play ${track.name}`}
                  className="relative w-11 h-11 shrink-0 rounded-lg overflow-hidden group/play disabled:cursor-default focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
                  style={!track.artwork_url ? { background: gradientFor(track.name) } : undefined}
                >
                  {track.artwork_url && (
                    <img src={track.artwork_url} alt="" className="w-full h-full object-cover" loading="lazy" />
                  )}
                  {track.preview_url && (
                    <span
                      className={`absolute inset-0 flex items-center justify-center bg-black/50 text-white transition-opacity ${
                        playing ? "opacity-100" : "opacity-0 group-hover/play:opacity-100"
                      }`}
                    >
                      {playing ? <PauseIcon /> : <PlayIcon />}
                    </span>
                  )}
                </button>
                <p className="flex-1 min-w-0 text-sm text-zinc-100 truncate">{track.name}</p>
                <button
                  type="button"
                  title={inAnyPlaylist ? "In your playlists" : "Add to playlist"}
                  aria-label={`${inAnyPlaylist ? "Manage" : "Add"} ${track.name} ${inAnyPlaylist ? "in" : "to"} your playlists`}
                  aria-pressed={inAnyPlaylist}
                  onClick={() => setOpenMenuTrackId((id) => (id === track.track_id ? null : track.track_id))}
                  className={`w-8 h-8 shrink-0 rounded-full flex items-center justify-center ring-1 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 ${
                    inAnyPlaylist
                      ? "bg-pink-500 text-white ring-pink-300/40"
                      : "bg-zinc-800 hover:bg-zinc-700 text-zinc-300 hover:text-pink-400 ring-white/10"
                  }`}
                >
                  <HeartIcon filled={inAnyPlaylist} />
                </button>
                {openMenuTrackId === track.track_id && (
                  <AddToPlaylistMenu
                    playlists={playlists}
                    isInPlaylist={(playlistId) =>
                      playlists.find((p) => p.id === playlistId)?.songs.some((s) => s.track_id === track.track_id)
                    }
                    onToggle={(playlistId, checked) =>
                      onTogglePlaylist(playlistId, { ...track, artist_name: artist.name }, checked)
                    }
                    onCreateAndAdd={(name) => {
                      onCreatePlaylistAndAdd(name, { ...track, artist_name: artist.name });
                      setOpenMenuTrackId(null);
                    }}
                    onClose={() => setOpenMenuTrackId(null)}
                  />
                )}
              </li>
            );
          })}
        </ul>
      )}
    </Modal>
  );
}
