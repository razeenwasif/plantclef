"use client";

import {
  ResponsiveContainer, ComposedChart, Bar, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, Cell, LabelList, Label,
} from "recharts";
import { UNFREEZE_SWEEP } from "@/lib/chart_data";
import { COLORS, AXIS_STYLE, LABEL_STYLE, TOOLTIP_STYLE } from "./theme";

export default function UnfreezeSweep() {
  const data = UNFREEZE_SWEEP;

  return (
    <div className="chart-card">
      <div className="chart-head">
        <div>
          <p className="chart-eyebrow">Figure 2</p>
          <h4>Val and Kaggle move in opposite directions</h4>
        </div>
        <p className="chart-cap">
          Holding every other knob fixed, sweeping the number of unfrozen
          transformer blocks. Val top 5 rises monotonically; Kaggle F1 peaks
          sharply at <b>n=4</b>. The bottleneck is train versus test
          distribution, not capacity.
        </p>
      </div>
      <div style={{ width: "100%", height: 360, minWidth: 0 }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 24, right: 8, left: 0, bottom: 24 }}>
            <CartesianGrid stroke={COLORS.hairlineSoft} strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="label"
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
            />
            <YAxis
              yAxisId="left"
              domain={[0.15, 0.45]}
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
              tickFormatter={(v) => v.toFixed(2)}
              width={36}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[0.6, 1.0]}
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={{ stroke: COLORS.hairline }}
              tickFormatter={(v) => v.toFixed(2)}
              width={36}
            />
            <Tooltip
              {...TOOLTIP_STYLE}
              formatter={(v, n) => [Number(v).toFixed(3), n === "kaggle" ? "Kaggle F1" : "val top 5"]}
            />
            <Legend
              verticalAlign="top"
              align="left"
              wrapperStyle={{ paddingBottom: 8, fontFamily: "Inter, sans-serif", fontSize: 12 }}
              formatter={(v) => v === "kaggle" ? "Kaggle F1" : "val top 5"}
            />
            <Bar
              yAxisId="left"
              dataKey="kaggle"
              fill={COLORS.ink}
              radius={[2, 2, 0, 0]}
              isAnimationActive
              animationDuration={900}
            >
              {data.map((d, i) => (
                <Cell
                  key={i}
                  fill={d.n === 4 ? COLORS.ink : COLORS.ink}
                  fillOpacity={d.n === 4 ? 1 : 0.78}
                  stroke={d.n === 4 ? COLORS.accent : "none"}
                  strokeWidth={d.n === 4 ? 1.5 : 0}
                />
              ))}
              <LabelList
                dataKey="kaggle"
                position="insideTop"
                offset={6}
                fill="#ffffff"
                style={{ fontFamily: "JetBrains Mono, monospace", fontSize: 11, fontWeight: 500 }}
                formatter={(v) => v.toFixed(3)}
              />
            </Bar>
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="val_top5"
              stroke={COLORS.accent}
              strokeWidth={2}
              dot={{ r: 3.5, fill: COLORS.accent, stroke: "none" }}
              activeDot={{ r: 5, fill: COLORS.accent }}
              isAnimationActive
              animationDuration={1100}
              animationBegin={300}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
