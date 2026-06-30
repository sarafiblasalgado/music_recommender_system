import { useEffect, useRef, useState } from "react";
import {
  listListeners, randomListenerId, getHomeFeed, sendFeedback,
  getPlaylists, createPlaylist, deletePlaylist, addSongToPlaylist, removeSongFromPlaylist,
} from "./api";
import ListenerSwitcher from "./components/ListenerSwitcher";
import ProfileHero from "./components/ProfileHero";
import Shelf from "./components/Shelf";
import HomeFeedSkeleton from "./components/HomeFeedSkeleton";
import ArtistTracksModal from "./components/ArtistTracksModal";
import PlaylistsPanel from "./components/PlaylistsPanel";
import MiniPlayer from "./components/MiniPlayer";

const REFETCH_DEBOUNCE_MS = 700;

function withoutSkipped(items, skippedIds) {
  if (!items) return items;
  return items.filter((a) => !skippedIds.has(a.id));
}

function App() {
  const [listeners, setListeners] = useState([]);
  const [currentId, setCurrentId] = useState(null);
  const [home, setHome] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [likedIds, setLikedIds] = useState(() => new Set());
  const [skippedIds, setSkippedIds] = useState(() => new Set());
  const [openArtist, setOpenArtist] = useState(null);
  const [playlists, setPlaylists] = useState([]);
  const [playlistsOpen, setPlaylistsOpen] = useState(false);
  const [playingTrack, setPlayingTrack] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const refetchTimer = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    listListeners().then(setListeners).catch(() => {});
    randomListenerId()
      .then(setCurrentId)
      .catch(() => setError("Couldn't reach the recommendation service."));
  }, []);

  useEffect(() => {
    if (currentId == null) return;
    setLoading(true);
    setError(null);
    setSkippedIds(new Set());
    setOpenArtist(null);
    setPlayingTrack(null);
    setIsPlaying(false);
    getHomeFeed(currentId)
      .then((data) => {
        setHome(data);
        setLikedIds(new Set(data.liked_artist_ids || []));
      })
      .catch(() => setError("Couldn't load this listener's feed."))
      .finally(() => setLoading(false));
    getPlaylists(currentId).then(setPlaylists).catch(() => setPlaylists([]));
  }, [currentId]);

  function scheduleRefetch() {
    if (refetchTimer.current) clearTimeout(refetchTimer.current);
    refetchTimer.current = setTimeout(() => {
      getHomeFeed(currentId).then((data) => {
        setHome(data);
        setLikedIds(new Set(data.liked_artist_ids || []));
        setSkippedIds(new Set());
      });
    }, REFETCH_DEBOUNCE_MS);
  }

  function handleLike(artistId) {
    setLikedIds((prev) => new Set(prev).add(artistId));
    sendFeedback(currentId, artistId, "like");
    scheduleRefetch();
  }

  function handleSkip(artistId) {
    setSkippedIds((prev) => new Set(prev).add(artistId));
    sendFeedback(currentId, artistId, "skip");
    scheduleRefetch();
  }

  function handleShuffle() {
    randomListenerId().then(setCurrentId);
  }

  function songFromTrack(track) {
    return {
      track_id: track.track_id,
      track_name: track.name,
      artist_id: openArtist?.id,
      artist_name: track.artist_name || openArtist?.name,
      artwork_url: track.artwork_url,
      preview_url: track.preview_url,
    };
  }

  function handleTogglePlaylist(playlistId, track, wasChecked) {
    if (wasChecked) {
      setPlaylists((prev) => prev.map((p) =>
        p.id === playlistId ? { ...p, songs: p.songs.filter((s) => s.track_id !== track.track_id) } : p
      ));
      removeSongFromPlaylist(currentId, playlistId, track.track_id);
    } else {
      const song = songFromTrack(track);
      setPlaylists((prev) => prev.map((p) =>
        p.id === playlistId ? { ...p, songs: [...p.songs, song] } : p
      ));
      addSongToPlaylist(currentId, playlistId, song.artist_id, track);
    }
  }

  function handleCreatePlaylistAndAdd(name, track) {
    createPlaylist(currentId, name).then(({ id }) => {
      const song = songFromTrack(track);
      setPlaylists((prev) => [...prev, { id, name, songs: [song] }]);
      addSongToPlaylist(currentId, id, song.artist_id, track);
    });
  }

  function handleCreatePlaylist(name) {
    createPlaylist(currentId, name).then(({ id }) => {
      setPlaylists((prev) => [...prev, { id, name, songs: [] }]);
    });
  }

  function handleDeletePlaylist(playlistId) {
    setPlaylists((prev) => prev.filter((p) => p.id !== playlistId));
    deletePlaylist(currentId, playlistId);
  }

  function handleRemoveSongFromPlaylist(playlistId, trackId) {
    setPlaylists((prev) => prev.map((p) =>
      p.id === playlistId ? { ...p, songs: p.songs.filter((s) => s.track_id !== trackId) } : p
    ));
    removeSongFromPlaylist(currentId, playlistId, trackId);
  }

  function handlePlayToggle(track) {
    if (playingTrack && playingTrack.track_id === track.track_id) {
      setIsPlaying((p) => !p);
    } else {
      setPlayingTrack(track);
      setIsPlaying(true);
      setProgress(0);
    }
  }

  function handleClosePlayer() {
    audioRef.current?.pause();
    setPlayingTrack(null);
    setIsPlaying(false);
  }

  function handleAudioTimeUpdate() {
    const audio = audioRef.current;
    if (audio && audio.duration) {
      setProgress(audio.currentTime / audio.duration);
    }
  }

  function handleAudioEnded() {
    setIsPlaying(false);
    setProgress(0);
  }

  const sections = home?.sections;

  return (
    <div className="min-h-screen bg-[#0a0a0c]">
      <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden="true">
        <div className="absolute -top-48 -left-32 w-[34rem] h-[34rem] rounded-full bg-violet-700/25 blur-[120px]" />
        <div className="absolute top-1/4 -right-40 w-[30rem] h-[30rem] rounded-full bg-pink-600/15 blur-[120px]" />
        <div className="absolute bottom-0 left-1/3 w-[26rem] h-[26rem] rounded-full bg-indigo-600/15 blur-[110px]" />
      </div>
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:top-3 focus:left-3 focus:bg-violet-600 focus:text-white focus:px-4 focus:py-2 focus:rounded-full"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-30 backdrop-blur bg-black/40 border-b border-white/5">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 md:px-10 py-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-400 to-pink-500" aria-hidden="true" />
            <span className="text-lg font-bold tracking-tight text-zinc-50">Wavelength</span>
          </div>
          <div className="flex items-center gap-3 flex-wrap justify-end">
            <button
              type="button"
              onClick={() => setPlaylistsOpen(true)}
              aria-label={`Open your playlists (${playlists.length} playlist${playlists.length === 1 ? "" : "s"})`}
              className="relative flex items-center gap-2 bg-zinc-800/80 hover:bg-zinc-700/80 text-sm text-zinc-100 rounded-full px-4 py-2 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400/60"
            >
              <svg viewBox="0 0 24 24" className="w-4 h-4" fill="currentColor" aria-hidden="true">
                <path d="M9 18V5l11-2v13M9 18a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm11-2a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Playlists
              {playlists.length > 0 && (
                <span className="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1 rounded-full bg-pink-500 text-white text-xs font-semibold">
                  {playlists.length}
                </span>
              )}
            </button>
            <ListenerSwitcher
              listeners={listeners}
              currentId={currentId}
              onSelect={setCurrentId}
              onShuffle={handleShuffle}
            />
          </div>
        </div>
      </header>

      <main id="main-content" className={`max-w-6xl mx-auto ${playingTrack ? "pb-32" : "pb-20"}`}>
        {error && (
          <div className="px-6 md:px-10 pt-10 text-zinc-400" role="alert">{error}</div>
        )}

        {!error && loading && <HomeFeedSkeleton />}

        {!error && !loading && home && sections && (
          <>
            <ProfileHero profile={home.profile} />
            <Shelf
              title="Made For You"
              accent="violet"
              items={withoutSkipped(sections.made_for_you, skippedIds)}
              likedIds={likedIds}
              onLike={handleLike}
              onSkip={handleSkip}
              onOpenTracks={setOpenArtist}
            />
            <Shelf
              title="Friends Are Listening To"
              subtitle="From people in your network"
              accent="teal"
              items={withoutSkipped(sections.friends_listening, skippedIds)}
              likedIds={likedIds}
              onLike={handleLike}
              onSkip={handleSkip}
              onOpenTracks={setOpenArtist}
            />
            {sections.because_you_listened && (
              <Shelf
                title={`Because you listened to ${sections.because_you_listened.seed_artist.name}`}
                accent="pink"
                items={withoutSkipped(sections.because_you_listened.items, skippedIds)}
                likedIds={likedIds}
                onLike={handleLike}
                onSkip={handleSkip}
                onOpenTracks={setOpenArtist}
              />
            )}
            <Shelf
              title="Trending Now"
              accent="amber"
              items={withoutSkipped(sections.trending, skippedIds)}
              likedIds={likedIds}
              onLike={handleLike}
              onSkip={handleSkip}
              onOpenTracks={setOpenArtist}
            />
          </>
        )}
      </main>

      {openArtist && (
        <ArtistTracksModal
          artist={openArtist}
          playlists={playlists}
          onTogglePlaylist={handleTogglePlaylist}
          onCreatePlaylistAndAdd={handleCreatePlaylistAndAdd}
          onClose={() => setOpenArtist(null)}
          currentTrackId={playingTrack?.track_id}
          isPlaying={isPlaying}
          onPlayToggle={handlePlayToggle}
        />
      )}
      {playlistsOpen && (
        <PlaylistsPanel
          playlists={playlists}
          onCreate={handleCreatePlaylist}
          onDelete={handleDeletePlaylist}
          onRemoveSong={handleRemoveSongFromPlaylist}
          onClose={() => setPlaylistsOpen(false)}
          currentTrackId={playingTrack?.track_id}
          isPlaying={isPlaying}
          onPlayToggle={handlePlayToggle}
        />
      )}
      <MiniPlayer
        track={playingTrack}
        isPlaying={isPlaying}
        progress={progress}
        onToggle={() => handlePlayToggle(playingTrack)}
        onClose={handleClosePlayer}
        onTimeUpdate={handleAudioTimeUpdate}
        onEnded={handleAudioEnded}
        audioRef={audioRef}
      />
    </div>
  );
}

export default App;
