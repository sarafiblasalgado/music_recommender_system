import { useEffect, useRef, useState } from "react";

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 13l4 4L19 7" />
    </svg>
  );
}

// A small popover anchored to a track's "add" button, listing the
// listener's playlists as toggleable checkboxes (a track can be in more
// than one) plus a quick "create new playlist and add" form. Closes on
// outside click or Escape.
export default function AddToPlaylistMenu({ playlists, isInPlaylist, onToggle, onCreateAndAdd, onClose }) {
  const [newName, setNewName] = useState("");
  const ref = useRef(null);

  useEffect(() => {
    function handleClick(e) {
      if (ref.current && !ref.current.contains(e.target)) onClose();
    }
    function handleKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  function handleSubmit(e) {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;
    onCreateAndAdd(name);
    setNewName("");
  }

  return (
    <div
      ref={ref}
      role="menu"
      aria-label="Add to playlist"
      className="absolute right-0 top-9 z-20 w-56 bg-zinc-800 rounded-xl ring-1 ring-white/10 shadow-2xl py-2 animate-fade-up-fast"
    >
      <p className="px-3 py-1 text-xs font-semibold text-zinc-400 uppercase tracking-wide">Add to playlist</p>
      {playlists.length > 0 && (
        <ul className="max-h-40 overflow-y-auto">
          {playlists.map((p) => {
            const checked = isInPlaylist(p.id);
            return (
              <li key={p.id}>
                <button
                  type="button"
                  role="menuitemcheckbox"
                  aria-checked={checked}
                  onClick={() => onToggle(p.id, checked)}
                  className="w-full flex items-center justify-between gap-2 px-3 py-1.5 text-sm text-left text-zinc-200 hover:bg-white/10 transition focus-visible:outline-none focus-visible:bg-white/10"
                >
                  <span className="truncate">{p.name}</span>
                  {checked && <span className="text-violet-400 shrink-0"><CheckIcon /></span>}
                </button>
              </li>
            );
          })}
        </ul>
      )}
      {playlists.length === 0 && (
        <p className="px-3 py-1.5 text-sm text-zinc-500">No playlists yet.</p>
      )}
      <form onSubmit={handleSubmit} className="flex gap-1.5 px-2 pt-2 mt-1 border-t border-white/5">
        <label htmlFor="new-playlist-name" className="sr-only">New playlist name</label>
        <input
          id="new-playlist-name"
          type="text"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="New playlist…"
          className="flex-1 min-w-0 bg-zinc-900 text-zinc-100 placeholder-zinc-500 text-xs rounded-lg px-2.5 py-1.5 outline-none focus:ring-2 focus:ring-violet-400/60"
        />
        <button
          type="submit"
          disabled={!newName.trim()}
          className="shrink-0 text-xs font-semibold px-2.5 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:hover:bg-violet-600 text-white transition"
        >
          Add
        </button>
      </form>
    </div>
  );
}
