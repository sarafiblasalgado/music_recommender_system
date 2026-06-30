import { useState } from "react";
import { gradientFor, hueFor } from "../avatarColor";

function HeartIcon({ filled }) {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2">
      <path d="M12 21s-7.5-4.6-10-9.1C0.3 8.2 1.8 4.5 5.3 3.6c2-.5 4 .3 5.2 2 .9-1.7 2.9-2.5 4.9-2 3.5.9 5 4.6 3.3 8.3C19.5 16.4 12 21 12 21z" />
    </svg>
  );
}

function SkipIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

export default function ArtistCard({ artist, size = "md", liked = false, onLike, onSkip, onOpenTracks }) {
  const [broken, setBroken] = useState(false);
  const showImage = artist.image_url && !broken;
  const dims = size === "sm" ? "w-28 h-28" : "w-40 h-40";
  const interactive = Boolean(onLike || onSkip);
  const hue = hueFor(artist.name || "?");

  return (
    <div className="shrink-0 w-40 select-none">
      <div className={`relative ${dims} mx-auto`}>
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -inset-4 rounded-full opacity-0 blur-xl transition-opacity duration-300 group-hover:opacity-60 group-focus-within:opacity-60"
          style={{ background: `hsl(${hue} 85% 55%)` }}
        />
        <button
          type="button"
          onClick={() => onOpenTracks?.(artist)}
          aria-label={`View top tracks for ${artist.name}`}
          className="block relative w-full h-full rounded-full overflow-hidden shadow-lg shadow-black/40 ring-1 ring-white/5 transition-all duration-300 ease-out group-hover:scale-110 group-hover:-translate-y-1 group-hover:shadow-2xl group-hover:ring-2 group-hover:ring-white/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
          style={!showImage ? { background: gradientFor(artist.name || "?") } : undefined}
        >
          {showImage ? (
            <img
              src={artist.image_url}
              alt=""
              className="w-full h-full object-cover"
              loading="lazy"
              onError={() => setBroken(true)}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-3xl font-semibold text-white/90">
              {(artist.name || "?").charAt(0).toUpperCase()}
            </div>
          )}
        </button>

        {interactive && (
          <div className="absolute inset-x-0 -bottom-1 flex justify-center gap-2 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-150">
            {onSkip && (
              <button
                type="button"
                title="Not interested"
                aria-label={`Not interested in ${artist.name}`}
                onClick={() => onSkip(artist.id)}
                className="w-8 h-8 rounded-full bg-zinc-900/90 hover:bg-zinc-800 text-zinc-300 hover:text-white flex items-center justify-center ring-1 ring-white/10 shadow-md transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
              >
                <SkipIcon />
              </button>
            )}
            {onLike && (
              <button
                type="button"
                title={liked ? "Liked" : "Like"}
                aria-label={`${liked ? "Liked" : "Like"} ${artist.name}`}
                aria-pressed={liked}
                onClick={() => onLike(artist.id)}
                className={`w-8 h-8 rounded-full flex items-center justify-center ring-1 shadow-md transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 ${
                  liked
                    ? "bg-pink-500 text-white ring-pink-300/40"
                    : "bg-zinc-900/90 hover:bg-zinc-800 text-zinc-300 hover:text-pink-400 ring-white/10"
                }`}
              >
                <HeartIcon filled={liked} />
              </button>
            )}
          </div>
        )}
      </div>
      <p className="mt-3 text-sm font-medium text-center text-zinc-100 truncate transition-colors duration-300 group-hover:text-white">
        {artist.name}
      </p>
      {artist.top_track && (
        <p className="text-xs text-center text-zinc-500 truncate">{artist.top_track}</p>
      )}
    </div>
  );
}
