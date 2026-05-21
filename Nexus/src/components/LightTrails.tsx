import './LightTrails.css';

// LightTrails — full-viewport diagonal light-streak background, intended
// as the global canvas for the Nexus shell. Drop it once at the top of
// App.tsx; it positions itself fixed at z-index -10 so the rest of the
// React tree paints over it without any layout interaction.
//
// The animation is pure CSS (see LightTrails.css). Each streak is a
// gradient-filled rect inside a rotated SVG group; CSS keyframes
// translate it horizontally, which the parent rotation turns into a
// diagonal top-right → bottom-left drift.

interface Streak {
  y: number;          // vertical offset within the 1600×900 viewBox
  length: number;     // streak length in viewBox px
  thickness: number;  // streak thickness
  opacity: number;    // base opacity multiplier (0..1)
  duration: number;   // animation duration in seconds (slower = longer trail)
  delay: number;      // negative delay = phase offset so streaks don't all start together
}

// Hand-tuned constellation. Kept deliberately subdued so the streaks
// read as atmosphere, not a focal element — content always wins. Max
// per-streak opacity capped at 0.35, durations slowed so motion is
// barely perceptible at a glance.
const STREAKS: Streak[] = [
  { y:  40, length: 1100, thickness: 1.0, opacity: 0.12, duration: 24, delay:  -2 },
  { y: 110, length:  900, thickness: 1.8, opacity: 0.28, duration: 16, delay:  -5 },
  { y: 180, length: 1400, thickness: 0.8, opacity: 0.10, duration: 28, delay:  -9 },
  { y: 250, length: 1000, thickness: 2.2, opacity: 0.35, duration: 13, delay:  -1 },
  { y: 330, length:  850, thickness: 1.2, opacity: 0.18, duration: 19, delay:  -7 },
  { y: 410, length: 1250, thickness: 1.5, opacity: 0.22, duration: 16, delay:  -3 },
  { y: 490, length:  700, thickness: 0.8, opacity: 0.12, duration: 22, delay: -11 },
  { y: 565, length: 1100, thickness: 1.8, opacity: 0.30, duration: 14, delay:  -6 },
  { y: 640, length:  900, thickness: 1.0, opacity: 0.16, duration: 20, delay:  -4 },
  { y: 715, length: 1200, thickness: 1.3, opacity: 0.22, duration: 15, delay:  -2 },
  { y: 790, length:  800, thickness: 0.8, opacity: 0.11, duration: 24, delay: -10 },
  { y: 855, length: 1050, thickness: 1.2, opacity: 0.18, duration: 18, delay:  -8 },
];

export const LightTrails = () => (
  <div className="lighttrails-bg" aria-hidden>
    <svg
      className="lighttrails-svg"
      viewBox="0 0 1600 900"
      preserveAspectRatio="xMidYMid slice"
    >
      <defs>
        {/* Streak gradient — fades at both ends, brightest just past
         * centre to give the asymmetric "light trail" feel. */}
        <linearGradient id="lt-streak" x1="0%" y1="50%" x2="100%" y2="50%">
          <stop offset="0%"   stopColor="white" stopOpacity="0" />
          <stop offset="40%"  stopColor="white" stopOpacity="0.45" />
          <stop offset="55%"  stopColor="white" stopOpacity="0.7" />
          <stop offset="72%"  stopColor="white" stopOpacity="0.35" />
          <stop offset="100%" stopColor="white" stopOpacity="0" />
        </linearGradient>

        {/* Gaussian blur softens the edges into the canvas. Filter
         * region is widened so the blur doesn't clip at the rect bbox. */}
        <filter id="lt-soften" x="-5%" y="-50%" width="115%" height="200%">
          <feGaussianBlur stdDeviation="0.9" />
        </filter>
      </defs>

      {/* Rotate the whole layer −22° so the streaks angle from top-right
       * to bottom-left. Anchor the rotation at the viewBox centre. */}
      <g transform="rotate(-22 800 450)" filter="url(#lt-soften)">
        {STREAKS.map((s, i) => (
          <rect
            key={i}
            className="lighttrails-streak"
            x={-s.length - 200}
            y={s.y}
            width={s.length}
            height={s.thickness}
            fill="url(#lt-streak)"
            opacity={s.opacity}
            style={{
              animationDuration: `${s.duration}s`,
              animationDelay: `${s.delay}s`,
            }}
          />
        ))}
      </g>
    </svg>
  </div>
);
