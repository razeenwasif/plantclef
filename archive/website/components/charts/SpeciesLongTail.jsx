"use client";

import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, Label,
} from "recharts";
import { SPECIES_LONG_TAIL } from "@/lib/chart_data";
import { COLORS, AXIS_STYLE, LABEL_STYLE, TOOLTIP_STYLE } from "./theme";

export default function SpeciesLongTail() {
  const d = SPECIES_LONG_TAIL;
  // Recharts wants array-of-objects, one per bin.
  const data = d.bins.map((bin, i) => ({
    bin,
    before: d.before_cap[i],
    after:  d.after_cap[i],
  }));

  return (
    <div className="chart-card">
      <div className="chart-head">
        <div>
          <p className="chart-eyebrow">Figure 5</p>
          <h4>Species image counts, before and after the 500 cap</h4>
        </div>
        <p className="chart-cap">
          PC24 is heavily right skewed. 2,263 species have 501 to 1,000
          training images while 145 have a single image. The 500 cap from
          i003 collapses the right tail but leaves <b>1,644 species with
          under 100 images</b> unchanged.
        </p>
      </div>
      <div style={{ width: "100%", height: 360, minWidth: 0 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 24, right: 8, left: 0, bottom: 32 }}>
            <CartesianGrid stroke={COLORS.hairlineSoft} strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="bin"
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
              interval={0}
            />
            <YAxis
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
              tickFormatter={(v) => v.toLocaleString()}
              width={44}
            />
            <Tooltip
              {...TOOLTIP_STYLE}
              formatter={(v, n) => [Number(v).toLocaleString(), n === "before" ? "i002 raw" : "i003 after 500 cap"]}
            />
            <Legend
              verticalAlign="top"
              align="left"
              wrapperStyle={{ paddingBottom: 8, fontFamily: "Inter, sans-serif", fontSize: 12 }}
              formatter={(v) => v === "before" ? "i002 raw distribution" : "i003 after 500 cap"}
            />
            <Bar
              dataKey="before"
              fill={COLORS.ink}
              fillOpacity={0.8}
              radius={[2, 2, 0, 0]}
              isAnimationActive
              animationDuration={900}
            />
            <Bar
              dataKey="after"
              fill={COLORS.accent}
              fillOpacity={0.85}
              radius={[2, 2, 0, 0]}
              isAnimationActive
              animationDuration={900}
              animationBegin={200}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
