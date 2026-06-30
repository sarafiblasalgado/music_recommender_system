import { useMemo, useRef, useState } from "react";

export default function ListenerSwitcher({ listeners, currentId, onSelect, onShuffle }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const pool = q
      ? listeners.filter(
          (l) =>
            String(l.id).includes(q) ||
            (l.top_artist && l.top_artist.toLowerCase().includes(q))
        )
      : listeners;
    return pool.slice(0, 30);
  }, [listeners, query]);

  function handleSelect(id) {
    onSelect(id);
    setQuery("");
    setOpen(false);
  }

  function handleKeyDown(e) {
    if (e.key === "Escape") {
      setOpen(false);
      e.currentTarget.blur();
    }
  }

  return (
    <div className="relative flex flex-wrap items-center justify-end gap-3" ref={containerRef}>
      <div className="relative">
        <label htmlFor="listener-search" className="sr-only">
          Search listeners by ID or top artist
        </label>
        <input
          id="listener-search"
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-controls="listener-search-results"
          aria-autocomplete="list"
          value={open ? query : ""}
          placeholder={`Listening as #${currentId ?? "..."}`}
          onFocus={() => setOpen(true)}
          onChange={(e) => setQuery(e.target.value)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={handleKeyDown}
          className="w-36 sm:w-64 bg-zinc-800/80 text-sm text-zinc-100 placeholder-zinc-400 rounded-full px-4 py-2 outline-none focus:ring-2 focus:ring-violet-400/60 transition"
        />
        {open && (
          <ul
            id="listener-search-results"
            role="listbox"
            aria-label="Listener results"
            className="absolute right-0 mt-2 w-72 max-h-80 overflow-y-auto rounded-xl bg-zinc-900 ring-1 ring-white/10 shadow-2xl z-20"
          >
            {filtered.length === 0 && (
              <li className="px-4 py-3 text-sm text-zinc-500">No listener matches "{query}"</li>
            )}
            {filtered.map((l) => (
              <li key={l.id} role="option" aria-selected={l.id === currentId}>
                <button
                  type="button"
                  onMouseDown={() => handleSelect(l.id)}
                  className="w-full text-left px-4 py-2.5 text-sm text-zinc-200 hover:bg-violet-500/20 transition flex justify-between focus-visible:outline-none focus-visible:bg-violet-500/20"
                >
                  <span>Listener #{l.id}</span>
                  {l.top_artist && (
                    <span className="text-zinc-500 truncate ml-3">{l.top_artist}</span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
      <button
        type="button"
        onClick={onShuffle}
        title="Switch to a random listener"
        aria-label="Switch to a random listener"
        className="bg-gradient-to-br from-violet-500 to-pink-500 hover:opacity-90 transition rounded-full px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-violet-900/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-300"
      >
        Shuffle
      </button>
    </div>
  );
}
