import { THEME_COLORS } from '../lib/site';

// RadialGauge — circular percentage indicator. Two concentric SVG arcs
// (track + filled portion). Stroke-dashoffset animates the fill on
// value change so the dial sweeps smoothly when the underlying metric
// updates.

interface RadialGaugeProps {
  value: number;          // 0..100
  label?: string;         // small caps label below the number
  size?: number;          // overall SVG box (square)
  strokeWidth?: number;
  color?: string;         // accent stroke colour
  trackColor?: string;    // unfilled portion colour
  unit?: string;          // appended to the number ("%", "°C", ...)
}

export const RadialGauge = ({
  value,
  label,
  size = 96,
  strokeWidth = 8,
  color = THEME_COLORS.accent,
  trackColor = 'rgba(255, 255, 255, 0.06)',
  unit = '%',
}: RadialGaugeProps) => {
  const v = Math.max(0, Math.min(100, value));
  const r = (size - strokeWidth) / 2;
  const c = size / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference - (v / 100) * circumference;

  return (
    <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={c} cy={c} r={r} fill="none" stroke={trackColor} strokeWidth={strokeWidth} />
        <circle
          cx={c}
          cy={c}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform={`rotate(-90 ${c} ${c})`}
          style={{ transition: 'stroke-dashoffset 600ms ease-out' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <span className="text-xl font-light text-white tracking-tighter leading-none">
          {v.toFixed(0)}
          {unit && (
            <span className="text-[10px] font-black text-oracle-muted uppercase opacity-60 ml-0.5">
              {unit}
            </span>
          )}
        </span>
        {label && (
          <span className="text-[8px] font-black text-oracle-muted uppercase tracking-widest opacity-50 mt-1">
            {label}
          </span>
        )}
      </div>
    </div>
  );
};
