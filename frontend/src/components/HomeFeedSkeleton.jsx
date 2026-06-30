function ShelfSkeleton() {
  return (
    <div className="mb-12">
      <div className="px-6 md:px-10 mb-4">
        <div className="h-6 w-48 rounded bg-white/10 animate-pulse" />
      </div>
      <div className="flex gap-6 overflow-x-hidden px-6 md:px-10">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="shrink-0 w-40 flex flex-col items-center">
            <div className="w-40 h-40 rounded-full bg-white/10 animate-pulse" />
            <div className="mt-3 h-4 w-24 rounded bg-white/10 animate-pulse" />
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HomeFeedSkeleton() {
  return (
    <div aria-busy="true" aria-live="polite">
      <span className="sr-only">Loading your recommendations&hellip;</span>
      <div className="px-6 md:px-10 pt-8 pb-10">
        <div className="h-3 w-28 rounded bg-white/10 animate-pulse mb-3" />
        <div className="h-9 w-64 rounded bg-white/10 animate-pulse mb-3" />
        <div className="h-4 w-96 max-w-full rounded bg-white/10 animate-pulse" />
      </div>
      <ShelfSkeleton />
      <ShelfSkeleton />
      <ShelfSkeleton />
    </div>
  );
}
