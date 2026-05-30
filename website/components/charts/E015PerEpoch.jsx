"use client";

import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from "recharts";
import { E015_PER_EPOCH } from "@/lib/chart_data";
import { COLORS, AXIS_STYLE, TOOLTIP_STYLE } from "./theme";

const SERIES = [
  { key: "top1",   label: "species top 1", color: COLORS.ink },
  { key: "top5",   label: "species top 5", color: COLORS.accent },
  { key: "genus",  label: "genus top 1",   color: COLORS.genus },
  { key: "family", label: "family top 1",  color: COLORS.family },
];

export default function E015PerEpoch() {
  const data = E015_PER_EPOCH.map((d) => ({ ...d, epochLabel: "ep " + d.epoch }));

  return (
    <div className="chart-card">
      <div className="chart-head">
        <div>
          <p className="chart-eyebrow">Figure 4</p>
          <h4>015 validation climbs while Kaggle stalls</h4>
        </div>
        <p className="chart-cap">
          Every taxonomic level rises smoothly across the 5 epoch PC24, iNat
          run. Yet Kaggle public F1 went from <b>0.37506</b> at ep1 to
          <b> 0.37956</b> at ep5, still below the <b>0.38333</b> 010 anchor
          without iNat. Adding capacity to the wrong distribution.
        </p>
      </div>
      <div style={{ width: "100%", height: 360, minWidth: 0 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 24, right: 16, left: 0, bottom: 56 }}>
            <CartesianGrid stroke={COLORS.hairlineSoft} strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="epochLabel"
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
            />
            <YAxis
              domain={[0.80, 0.97]}
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
              tickFormatter={(v) => v.toFixed(2)}
              width={36}
            />
            <Tooltip
              {...TOOLTIP_STYLE}
              formatter={(v, n) => {
                const s = SERIES.find((x) => x.key === n);
                return [Number(v).toFixed(4), s ? s.label : n];
              }}
            />
            <Legend
              verticalAlign="top"
              align="right"
              wrapperStyle={{ paddingBottom: 8, fontFamily: "Inter, sans-serif", fontSize: 12 }}
              formatter={(v) => SERIES.find((s) => s.key === v)?.label ?? v}
            />
            {SERIES.map((s) => (
              <Line
                key={s.key}
                type="monotone"
                dataKey={s.key}
                stroke={s.color}
                strokeWidth={s.key === "top1" || s.key === "top5" ? 2 : 1.4}
                dot={{ r: 3, fill: s.color }}
                activeDot={{ r: 5, fill: s.color }}
                isAnimationActive
                animationDuration={900}
              />
            ))}
            {/* The ep1/ep5 Kaggle scores are quoted directly in the chart
                caption above; we previously had reference-line annotations
                here, but they clipped on narrow viewports without adding
                anything the caption did not already say. */}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
