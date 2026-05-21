// Shared editorial color tokens for Recharts.

export const COLORS = {
  ink:        "#0c0a09",
  accent:     "#b45309",
  muted:      "#777169",
  mutedSoft:  "#a8a29e",
  hairline:   "#d6d3d1",
  hairlineSoft: "#e7e5e4",
  best:       "#b45309",
  cap:        "#1d4ed8",
  sweep:      "#6b21a8",
  inat:       "#b91c1c",
  ssl:        "#166534",
  genus:      "#166534",
  family:     "#6b21a8",
};

export const AXIS_STYLE = {
  fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
  fontSize: 12,
  fill: COLORS.muted,
};

export const LABEL_STYLE = {
  fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, sans-serif',
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: "0.6px",
  textTransform: "uppercase",
  fill: COLORS.muted,
};

export const TOOLTIP_STYLE = {
  contentStyle: {
    background: "#ffffff",
    border: "1px solid " + COLORS.hairline,
    borderRadius: 8,
    boxShadow: "0 6px 24px rgba(0,0,0,0.08)",
    fontFamily: '"JetBrains Mono", ui-monospace, monospace',
    fontSize: 12,
    color: COLORS.ink,
    padding: "10px 12px",
  },
  labelStyle: {
    fontFamily: '"Inter", sans-serif',
    fontWeight: 600,
    fontSize: 11,
    color: COLORS.muted,
    textTransform: "uppercase",
    letterSpacing: "0.6px",
    marginBottom: 6,
  },
  itemStyle: {
    color: COLORS.ink,
    padding: "2px 0",
  },
  cursor: { stroke: COLORS.hairline, strokeWidth: 1, strokeDasharray: "3 4" },
};
