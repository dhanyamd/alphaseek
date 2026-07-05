"use client";

// Monochrome equity curve — clean white line over a faint grid.
export default function EquityChart({ data, height = 150 }: { data: number[]; height?: number }) {
  if (!data || data.length < 2) {
    return <div className="text-[11px] text-faint py-8 text-center mono">no data</div>;
  }
  const w = 500;
  const h = height;
  const pad = 6;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const rng = max - min || 1;
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - 2 * pad);
    const y = h - pad - ((v - min) / rng) * (h - 2 * pad);
    return [x, y];
  });
  const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const area = `${path} L${pts[pts.length - 1][0].toFixed(1)},${h - pad} L${pts[0][0].toFixed(1)},${h - pad} Z`;
  const last = pts[pts.length - 1];
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#ffffff" stopOpacity="0.10" />
          <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0.25, 0.5, 0.75].map((f) => (
        <line key={f} x1={pad} x2={w - pad} y1={h * f} y2={h * f} stroke="#1c1c1e" strokeWidth="1" />
      ))}
      <path d={area} fill="url(#eqfill)" />
      <path d={path} fill="none" stroke="#fafafa" strokeWidth="1.4" />
      <circle cx={last[0]} cy={last[1]} r="2.4" fill="#fafafa" />
    </svg>
  );
}
