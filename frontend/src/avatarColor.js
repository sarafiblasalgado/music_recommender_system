// Deterministic hue per artist name -- shared by the fallback avatar
// gradient and the hover glow, so a given artist always reads as "the
// same color" across the UI.
export function hueFor(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return Math.abs(hash) % 360;
}

// Deterministic gradient per artist name, used as the placeholder/fallback
// whenever a real photo isn't available (no iTunes match, or it fails to
// load) -- same idea as Slack/Spotify's generated avatars, so a missing
// image never looks like a broken page.
export function gradientFor(name) {
  const hue1 = hueFor(name);
  const hue2 = (hue1 + 55) % 360;
  return `linear-gradient(135deg, hsl(${hue1} 70% 45%), hsl(${hue2} 70% 30%))`;
}
