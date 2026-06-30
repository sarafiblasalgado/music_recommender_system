import { gradientFor } from "../avatarColor";

export default function ProfileHero({ profile }) {
  if (!profile) return null;
  const names = profile.top_artists.slice(0, 5).map((a) => a.name);
  const heroArtist = profile.top_artists[0];

  return (
    <div className="relative px-6 md:px-10 pt-16 pb-12 md:pt-24 md:pb-16 overflow-hidden">
      <div className="absolute inset-0 -z-10" aria-hidden="true">
        {heroArtist?.image_url ? (
          <img
            src={heroArtist.image_url}
            alt=""
            className="w-full h-full object-cover scale-110 blur-2xl opacity-40"
          />
        ) : (
          <div className="w-full h-full" style={{ background: gradientFor(heroArtist?.name || "?") }} />
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-[#0a0a0c] via-[#0a0a0c]/75 to-[#0a0a0c]/10" />
      </div>
      <p className="text-sm uppercase tracking-widest text-violet-300/80 font-semibold mb-2 animate-fade-up">
        Welcome back
      </p>
      <h1 className="text-4xl md:text-5xl font-extrabold text-zinc-50 mb-3 tracking-tight animate-fade-up" style={{ animationDelay: "60ms" }}>
        Listener #{profile.id}
      </h1>
      <p
        className="text-zinc-300 max-w-full text-base md:text-lg truncate animate-fade-up"
        style={{ animationDelay: "120ms" }}
        title={`Big into ${names.slice(0, 3).join(", ")}${names.length > 3 ? ` and ${names.length - 3} more` : ""}.${profile.n_friends > 0 ? ` Connected with ${profile.n_friends} friend${profile.n_friends === 1 ? "" : "s"} on the network.` : ""}`}
      >
        Big into {names.slice(0, 3).join(", ")}
        {names.length > 3 ? ` and ${names.length - 3} more` : ""}.
        {profile.n_friends > 0
          ? ` Connected with ${profile.n_friends} friend${profile.n_friends === 1 ? "" : "s"} on the network.`
          : ""}
      </p>
    </div>
  );
}
