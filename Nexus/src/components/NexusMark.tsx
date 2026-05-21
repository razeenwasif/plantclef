// NexusMark — brand glyph for the Nexus shell. Two diagonal trail
// strokes (echoing the LightTrails canvas) replace the Lucide Zap that
// appears on the Oracle product chip. Stroke colour follows
// `currentColor` so it inherits from the parent text class.

export const NexusMark = ({ size = 16 }: { size?: number }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    aria-hidden
  >
    <line
      x1="5" y1="19" x2="19" y2="5"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      opacity="0.95"
    />
    <line
      x1="10" y1="19" x2="19" y2="10"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      opacity="0.55"
    />
  </svg>
);
