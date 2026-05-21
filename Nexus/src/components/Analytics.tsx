import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import type { Run } from '../lib/runs';
import { THEME_COLORS } from '../lib/site';

// One distinct stroke per overlaid run. Cycled if there are more runs
// than colors — usually we only render a handful of in-flight runs.
// Source-of-truth lives in lib/site.ts so the rotation swaps to a
// monochrome palette automatically on the Nexus shell.
const SERIES_COLORS = THEME_COLORS.chartSeries;

interface AnalyticsChartProps {
  runs: Run[];                  // active + recent that have at least one step point
  metricKey?: 'loss' | 'local_acc';
}

/** Build Recharts-ready rows: one object per step, with one keyed column
 *  per run (so multiple series share the same x-axis if their step
 *  values line up). When step values diverge between runs (different
 *  start times, sparse points), Recharts still draws the lines
 *  independently because the missing keys are undefined per row. */
const buildRows = (runs: Run[], metric: 'loss' | 'local_acc'): Record<string, number | string>[] => {
  const allSteps = new Set<number>();
  for (const r of runs) for (const s of r.metrics.step) allSteps.add(s);
  const sorted = Array.from(allSteps).sort((a, b) => a - b);
  return sorted.map((step) => {
    const row: Record<string, number | string> = { step };
    for (const r of runs) {
      const idx = r.metrics.step.indexOf(step);
      if (idx === -1) continue;
      const v = metric === 'loss' ? r.metrics.loss[idx] : r.metrics.local_acc[idx];
      if (typeof v === 'number') row[r.id] = v;
    }
    return row;
  });
};

const labelFor = (run: Run): string => {
  const name = run.spec.name || run.id;
  // Trim long stems to keep the legend tidy
  return name.length > 28 ? `${name.slice(0, 25)}…` : name;
};

export const AnalyticsChart = ({ runs, metricKey = 'loss' }: AnalyticsChartProps) => {
  const chartable = runs.filter((r) => r.metrics.step.length > 1);
  if (chartable.length === 0) {
    return (
      <div className="w-full h-full min-h-[300px] flex items-center justify-center">
        <div className="text-center px-6">
          <p className="text-sm text-white/85 font-medium mb-1">No metrics yet</p>
          <p className="text-[11px] text-oracle-muted max-w-md leading-relaxed">
            Loss and validation curves stream in as soon as a run emits its first telemetry
            <code className="text-oracle-cyan font-mono px-1"> step </code>
            events. Submit a run from Fleet or via ⌘K.
          </p>
        </div>
      </div>
    );
  }
  const rows = buildRows(chartable, metricKey);

  return (
    <div className="w-full h-full min-h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 10, right: 16, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="step"
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#64748b', fontSize: 10, fontWeight: 700 }}
            label={{ value: 'step', position: 'insideBottomRight', offset: -2, fill: '#475569', fontSize: 9 }}
          />
          <YAxis
            axisLine={false}
            tickLine={false}
            tick={{ fill: '#64748b', fontSize: 10, fontWeight: 700 }}
            domain={['auto', 'auto']}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(10, 10, 18, 0.92)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '12px',
              fontSize: '11px',
              boxShadow: '0 10px 30px rgba(0,0,0,0.55)',
            }}
            labelStyle={{ color: '#94a3b8', fontWeight: 700, marginBottom: 4 }}
          />
          <Legend
            wrapperStyle={{ fontSize: 10, color: '#94a3b8' }}
            iconType="line"
            iconSize={12}
          />
          {chartable.map((run, i) => (
            <Line
              key={run.id}
              type="monotone"
              dataKey={run.id}
              name={labelFor(run)}
              stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
              strokeWidth={1.8}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
