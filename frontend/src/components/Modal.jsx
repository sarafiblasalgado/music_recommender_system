import { useEffect, useRef } from "react";

// Shared overlay/close/Escape mechanics for ArtistTracksModal and
// PlaylistPanel -- pulled out because getting focus + Escape + backdrop
// click right is fiddly enough that duplicating it risks the two drifting
// out of sync, not because either modal needed much else from the other.
export default function Modal({ title, onClose, onBack, children }) {
  const closeRef = useRef(null);

  useEffect(() => {
    closeRef.current?.focus();
    function handleKeyDown(e) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-40 flex items-start sm:items-center justify-center bg-black/70 backdrop-blur-sm px-4 py-8 sm:py-4 overflow-y-auto animate-backdrop-fade"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="w-full max-w-lg bg-zinc-900 rounded-2xl ring-1 ring-white/10 shadow-2xl my-auto animate-modal-pop"
      >
        <div className="flex items-center gap-2 px-5 py-4 border-b border-white/5">
          {onBack && (
            <button
              type="button"
              onClick={onBack}
              aria-label="Back"
              className="w-8 h-8 -ml-1 shrink-0 rounded-full flex items-center justify-center text-zinc-400 hover:text-white hover:bg-white/10 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
            >
              <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M15 18l-6-6 6-6" />
              </svg>
            </button>
          )}
          <h2 className="flex-1 min-w-0 text-lg font-bold text-zinc-50 truncate">{title}</h2>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="w-8 h-8 shrink-0 rounded-full flex items-center justify-center text-zinc-400 hover:text-white hover:bg-white/10 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
          >
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <path d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>
        <div className="max-h-[70vh] overflow-y-auto px-2 py-2">{children}</div>
      </div>
    </div>
  );
}
