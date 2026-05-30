// Shared SVG primitives for the five charts.

export function tickValues(min, max, n = 5) {
  const range = max - min;
  if (range <= 0) return [min];
  const step = range / n;
  const out = [];
  for (let i = 0; i <= n; i++) out.push(+(min + i * step).toFixed(3));
  return out;
}

export function YAxis({
  x, y0, y1, vMin, vMax, ticks = 5, label, labelOffset = 44,
  format = (v) => v.toFixed(2),
}) {
  const vals = tickValues(vMin, vMax, ticks);
  return (
    <g className="chart-axis chart-axis-y">
      <line x1={x} x2={x} y1={y0} y2={y1} />
      {vals.map((v, i) => {
        const ty = y0 + (1 - (v - vMin) / (vMax - vMin)) * (y1 - y0);
        return (
          <g key={i}>
            <line x1={x - 4} x2={x} y1={ty} y2={ty} />
            <text x={x - 8} y={ty} dy="0.32em" textAnchor="end">{format(v)}</text>
          </g>
        );
      })}
      {label && (
        <text
          className="chart-axis-label"
          x={x - labelOffset}
          y={(y0 + y1) / 2}
          textAnchor="middle"
          transform={`rotate(-90 ${x - labelOffset} ${(y0 + y1) / 2})`}
        >
          {label}
        </text>
      )}
    </g>
  );
}

export function YAxisRight({
  x, y0, y1, vMin, vMax, ticks = 5, label, format = (v) => v.toFixed(2),
}) {
  const vals = tickValues(vMin, vMax, ticks);
  return (
    <g className="chart-axis chart-axis-y chart-axis-y-right">
      <line x1={x} x2={x} y1={y0} y2={y1} />
      {vals.map((v, i) => {
        const ty = y0 + (1 - (v - vMin) / (vMax - vMin)) * (y1 - y0);
        return (
          <g key={i}>
            <line x1={x} x2={x + 4} y1={ty} y2={ty} />
            <text x={x + 8} y={ty} dy="0.32em" textAnchor="start">{format(v)}</text>
          </g>
        );
      })}
      {label && (
        <text
          className="chart-axis-label"
          x={x + 40}
          y={(y0 + y1) / 2}
          transform={`rotate(90 ${x + 40} ${(y0 + y1) / 2})`}
        >
          {label}
        </text>
      )}
    </g>
  );
}

export function XAxis({ x0, x1, y, ticks, label, vertical = false }) {
  return (
    <g className="chart-axis chart-axis-x">
      <line x1={x0} x2={x1} y1={y} y2={y} />
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={t.x} x2={t.x} y1={y} y2={y + 4} />
          <text
            x={t.x}
            y={y + 16}
            textAnchor={vertical ? "end" : "middle"}
            transform={vertical ? `rotate(-40 ${t.x} ${y + 16})` : ""}
          >
            {t.label}
          </text>
        </g>
      ))}
      {label && (
        <text className="chart-axis-label" x={(x0 + x1) / 2} y={y + 44} textAnchor="middle">
          {label}
        </text>
      )}
    </g>
  );
}

export function Gridlines({ x0, x1, y0, y1, vMin, vMax, ticks = 5 }) {
  const vals = tickValues(vMin, vMax, ticks);
  return (
    <g className="chart-grid">
      {vals.map((v, i) => {
        const ty = y0 + (1 - (v - vMin) / (vMax - vMin)) * (y1 - y0);
        return <line key={i} x1={x0} x2={x1} y1={ty} y2={ty} />;
      })}
    </g>
  );
}
