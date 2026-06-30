import { useState } from "react";
import Modal from "./Modal";
import { gradientFor } from "../avatarColor";

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

function RemoveIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

function PlaylistIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 18V5l11-2v13M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm11-2a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
    </svg>
  );
}

// The listener's own playlists -- a list view (create / open / delete)
// and, once one is selected, a detail view of its songs (remove / play),
// reusing the same Modal for both via its onBack affordance. A track can
// belong to several playlists at once (see AddToPlaylistMenu), so this is
// deliberately not a single flat collection.
export default function PlaylistsPanel({
  playlists, onCreate, onDelete, onRemoveSong, onClose,
  currentTrackId, isPlaying, onPlayToggle,
}) {
  const [selectedId, setSelectedId] = useState(null);
  const [newName, setNewName] = useState("");
  const selected = playlists.find((p) => p.id === selectedId);

  function handleCreate(e) {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;
    onCreate(name);
    setNewName("");
  }

  if (selected) {
    return (
      <Modal title={selected.name} onClose={onClose} onBack={() => setSelectedId(null)}>
        {selected.songs.length === 0 && (
          <div className="px-3 py-8 text-center text-sm text-zinc-500">
            No songs yet — click an artist card, then add a track here.
          </div>
        )}
        {selected.songs.length > 0 && (
          <ul className="divide-y divide-white/5">
            {selected.songs.map((song, i) => {
              const playing = currentTrackId === song.track_id && isPlaying;
              return (
                <li
                  key={song.track_id}
                  className="flex items-center gap-3 px-3 py-2.5 animate-fade-up"
                  style={{ animationDelay: `${Math.min(i * 30, 240)}ms` }}
                >
                  <button
                    type="button"
                    onClick={() => song.preview_url && onPlayToggle(song)}
                    disabled={!song.preview_url}
                    aria-label={playing ? `Pause ${song.track_name}` : `Play ${song.track_name}`}
                    className="relative w-11 h-11 shrink-0 rounded-lg overflow-hidden group/play disabled:cursor-default focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
                    style={!song.artwork_url ? { background: gradientFor(song.track_name) } : undefined}
                  >
                    {song.artwork_url && (
                      <img src={song.artwork_url} alt="" className="w-full h-full object-cover" loading="lazy" />
                    )}
                    {song.preview_url && (
                      <span
                        className={`absolute inset-0 flex items-center justify-center bg-black/50 text-white transition-opacity ${
                          playing ? "opacity-100" : "opacity-0 group-hover/play:opacity-100"
                        }`}
                      >
                        {playing ? <PauseIcon /> : <PlayIcon />}
                      </span>
                    )}
                  </button>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-zinc-100 truncate">{song.track_name}</p>
                    <p className="text-xs text-zinc-500 truncate">{song.artist_name}</p>
                  </div>
                  <button
                    type="button"
                    title="Remove from playlist"
                    aria-label={`Remove ${song.track_name} from ${selected.name}`}
                    onClick={() => onRemoveSong(selected.id, song.track_id)}
                    className="w-8 h-8 shrink-0 rounded-full flex items-center justify-center text-zinc-400 hover:text-white hover:bg-white/10 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
                  >
                    <RemoveIcon />
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        <div className="px-3 pt-3 pb-1">
          <button
            type="button"
            onClick={() => {
              onDelete(selected.id);
              setSelectedId(null);
            }}
            className="text-xs text-zinc-500 hover:text-red-400 transition"
          >
            Delete playlist
          </button>
        </div>
      </Modal>
    );
  }

  return (
    <Modal title="Your playlists" onClose={onClose}>
      {playlists.length === 0 && (
        <div className="px-3 py-6 text-center text-sm text-zinc-500">
          No playlists yet — click an artist card, then add a track to create one.
        </div>
      )}
      {playlists.length > 0 && (
        <ul className="divide-y divide-white/5">
          {playlists.map((p, i) => (
            <li key={p.id} className="animate-fade-up" style={{ animationDelay: `${Math.min(i * 40, 240)}ms` }}>
              <button
                type="button"
                onClick={() => setSelectedId(p.id)}
                className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-white/5 transition focus-visible:outline-none focus-visible:bg-white/5"
              >
                <span className="w-11 h-11 shrink-0 rounded-lg bg-zinc-800 flex items-center justify-center text-zinc-400">
                  <PlaylistIcon />
                </span>
                <span className="flex-1 min-w-0">
                  <span className="block text-sm text-zinc-100 truncate">{p.name}</span>
                  <span className="block text-xs text-zinc-500">{p.songs.length} song{p.songs.length === 1 ? "" : "s"}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
      <form onSubmit={handleCreate} className="flex gap-2 px-3 pt-3 mt-1 border-t border-white/5">
        <label htmlFor="create-playlist-name" className="sr-only">New playlist name</label>
        <input
          id="create-playlist-name"
          type="text"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="New playlist name…"
          className="flex-1 min-w-0 bg-zinc-800 text-zinc-100 placeholder-zinc-500 text-sm rounded-lg px-3 py-2 outline-none focus:ring-2 focus:ring-violet-400/60"
        />
        <button
          type="submit"
          disabled={!newName.trim()}
          className="shrink-0 text-sm font-semibold px-3 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:hover:bg-violet-600 text-white transition"
        >
          Create
        </button>
      </form>
    </Modal>
  );
}
