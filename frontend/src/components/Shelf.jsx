import ArtistCard from "./ArtistCard";

// Each shelf gets its own accent so the four rows read as distinct sections
// rather than four identical lists -- a colored rail by the title and a
// faint matching wash behind it.
const ACCENTS = {
  violet: { bar: "from-violet-400 to-violet-600", glow: "bg-violet-500/10" },
  teal: { bar: "from-teal-300 to-teal-500", glow: "bg-teal-500/10" },
  pink: { bar: "from-pink-400 to-pink-600", glow: "bg-pink-500/10" },
  amber: { bar: "from-amber-300 to-amber-500", glow: "bg-amber-500/10" },
};

export default function Shelf({ title, subtitle, items, likedIds, onLike, onSkip, onOpenTracks, accent = "violet" }) {
  if (!items || items.length === 0) return null;
  const theme = ACCENTS[accent] || ACCENTS.violet;

  return (
    <section className="relative mb-12">
      <div
        className={`pointer-events-none absolute -left-16 -top-10 w-72 h-72 rounded-full blur-3xl -z-10 ${theme.glow}`}
        aria-hidden="true"
      />
      <div className="px-6 md:px-10 mb-4 flex items-center gap-3 animate-fade-up">
        <span className={`h-6 w-1.5 rounded-full bg-gradient-to-b ${theme.bar}`} aria-hidden="true" />
        <div>
          <h2 className="text-xl md:text-2xl font-bold text-zinc-50">{title}</h2>
          {subtitle && <p className="text-sm text-zinc-400 mt-1">{subtitle}</p>}
        </div>
      </div>
      <div className="shelf-scroll flex gap-6 overflow-x-auto px-6 md:px-10 pb-2">
        {items.map((artist, i) => (
          <div
            key={artist.id}
            className="group animate-fade-up"
            style={{ animationDelay: `${Math.min(i * 50, 400)}ms` }}
          >
            <ArtistCard
              artist={artist}
              liked={likedIds ? likedIds.has(artist.id) : false}
              onLike={onLike}
              onSkip={onSkip}
              onOpenTracks={onOpenTracks}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
