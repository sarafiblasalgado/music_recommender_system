import { useEffect, useRef } from "react";
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

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

// A persistent bottom bar for the 30s Deezer preview clips (see
// ArtistTracksModal / PlaylistPanel for where playback is triggered).
// Deliberately a single shared <audio> element owned here, driven by
// props from App.jsx, rather than one per track row -- so playback
// survives closing the modal that started it, the way a real player would.
export default function MiniPlayer({ track, isPlaying, progress, onToggle, onClose, onTimeUpdate, onEnded, audioRef }) {
  const barRef = useRef(null);

  useEffect(() => {
    if (!track) return;
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.src !== track.preview_url) {
      audio.src = track.preview_url || "";
      audio.currentTime = 0;
    }
    if (isPlaying) {
      audio.play().catch(() => {});
    } else {
      audio.pause();
    }
  }, [track, isPlaying, audioRef]);

  function handleSeek(e) {
    const audio = audioRef.current;
    if (!audio || !audio.duration || !barRef.current) return;
    const rect = barRef.current.getBoundingClientRect();
    const ratio = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 1);
    audio.currentTime = ratio * audio.duration;
  }

  if (!track) return null;

  const name = track.track_name || track.name;

  return (
    <div className="fixed bottom-0 inset-x-0 z-50 bg-zinc-900/95 backdrop-blur-md border-t border-white/10 animate-fade-up-fast">
      <audio
        ref={audioRef}
        onTimeUpdate={onTimeUpdate}
        onEnded={onEnded}
      />
      <div
        ref={barRef}
        onClick={handleSeek}
        className="h-1 w-full bg-white/10 cursor-pointer group/seek"
      >
        <div
          className="h-full bg-gradient-to-r from-violet-400 to-pink-400 group-hover/seek:brightness-110"
          style={{ width: `${Math.min(progress * 100, 100)}%` }}
        />
      </div>
      <div className="max-w-6xl mx-auto px-4 sm:px-6 md:px-10 py-3 flex items-center gap-3">
        <div
          className="w-10 h-10 rounded-lg overflow-hidden shrink-0"
          style={!track.artwork_url ? { background: gradientFor(name || "?") } : undefined}
        >
          {track.artwork_url && <img src={track.artwork_url} alt="" className="w-full h-full object-cover" />}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm text-zinc-100 truncate">{name}</p>
          {track.artist_name && <p className="text-xs text-zinc-500 truncate">{track.artist_name}</p>}
        </div>
        <button
          type="button"
          onClick={onToggle}
          aria-label={isPlaying ? "Pause preview" : "Play preview"}
          className="w-9 h-9 shrink-0 rounded-full bg-white text-black flex items-center justify-center hover:scale-105 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
        >
          {isPlaying ? <PauseIcon /> : <PlayIcon />}
        </button>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close player"
          className="w-8 h-8 shrink-0 rounded-full text-zinc-400 hover:text-white hover:bg-white/10 flex items-center justify-center transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
        >
          <CloseIcon />
        </button>
      </div>
    </div>
  );
}
