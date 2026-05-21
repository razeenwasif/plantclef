import { ResponsiveContainer, LineChart, Line, YAxis } from 'recharts';
import { THEME_COLORS } from '../lib/site';

// Sparkline — thin inline line chart with no axes, no tooltip, no grid.
// Used inside StatCards to show the trend behind a single-number KPI;
// styled to match the surrounding card rather than to be read precisely.
//
// Returns null for series shorter than 2 points (nothing to draw).

interface SparklineProps {
  series: number[];
  height?: number;
  color?: string;
  strokeWidth?: number;
  /** Domain hint — pass 'auto' for [dataMin, dataMax], 'zero' for [0, dataMax]. */
  baseline?: 'auto' | 'zero';
}

export const Sparkline = ({
  series,
  height = 32,
  color = THEME_COLORS.chartPrimary,
  strokeWidth = 1.3,
  baseline = 'auto',
}: SparklineProps) => {
  if (!series || series.length < 2) return null;
  const data = series.map((y, x) => ({ x, y }));
  const domain: ['auto' | 'dataMin' | number, 'auto' | 'dataMax' | number] =
    baseline === 'zero' ? [0, 'dataMax'] : ['dataMin', 'dataMax'];

  return (
    <div style={{ height, width: '100%' }} aria-hidden>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
          <Line
            type="monotone"
            dataKey="y"
            stroke={color}
            strokeWidth={strokeWidth}
            dot={false}
            isAnimationActive={false}
          />
          <YAxis hide domain={domain} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
