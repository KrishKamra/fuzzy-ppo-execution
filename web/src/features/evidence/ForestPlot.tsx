import { clsx } from "clsx";
import { useId } from "react";

export interface ForestRow {
  key: string;
  label: string;
  sublabel?: string;
  mean: number;
  lo: number;
  hi: number;
  dateLo?: number | null;
  dateHi?: number | null;
  color: string;
  note?: string;
}

const fmt = (x: number) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(2)}`;

function ticks(lo: number, hi: number): number[] {
  const span = hi - lo;
  const step = [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2].find((s) => span / s <= 6) ?? 1;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

/**
 * Point estimates with 95% CIs: thick = session clusters, thin = date clusters (the slide figures use the same
 * encoding). The dashed zero line is the reference (VWAP, or the policy compared against). Every row also has a
 * text equivalent for screen readers.
 */
export function ForestPlot({ rows, axisLabel, reference = "VWAP" }: { rows: ForestRow[]; axisLabel: string; reference?: string }) {
  const id = useId();
  const values = rows.flatMap((r) => [r.lo, r.hi, r.dateLo ?? r.lo, r.dateHi ?? r.hi, 0]);
  const pad = (Math.max(...values) - Math.min(...values)) * 0.08 || 0.1;
  const lo = Math.min(...values) - pad;
  const hi = Math.max(...values) + pad;
  const rowH = 34;
  const H = rows.length * rowH; // tick labels overflow below, into the caption's top margin
  // Fixed pixel rows (aligned with the labels) and percentage x, so the plot stretches without distortion.
  const x = (v: number) => `${(((v - lo) / (hi - lo)) * 100).toFixed(3)}%`;
  return (
    <div className="grid grid-cols-[minmax(9rem,auto)_minmax(0,1fr)_minmax(4.5rem,auto)] gap-x-3"
      style={{ gridTemplateRows: `repeat(${rows.length}, ${rowH}px) auto` }}>
      <ul className="contents" aria-labelledby={`${id}-cap`}>
        {rows.map((r, i) => (
          <li key={r.key} className="contents">
            <span className="flex flex-col justify-center text-sm" style={{ gridRow: i + 1, height: rowH }}>
              <span className="flex items-center gap-2">
                <span aria-hidden className="size-2 shrink-0 rounded-full" style={{ background: r.color }} />
                <span className="truncate">{r.label}</span>
              </span>
              {r.sublabel && <span className="pl-4 text-[0.7rem] text-fg-subtle">{r.sublabel}</span>}
            </span>
            <span className="sr-only">
              {r.label} {r.sublabel ?? ""}: {fmt(r.mean)} basis points vs {reference}, 95% interval {fmt(r.lo)} to {fmt(r.hi)}. {r.note ?? ""}
            </span>
            <span className={clsx("tabular flex items-center justify-end font-mono text-xs text-fg-subtle")} style={{ gridRow: i + 1, gridColumn: 3, height: rowH }}>
              {r.note}
            </span>
          </li>
        ))}
      </ul>
      <svg width="100%" height={H} className="overflow-visible" style={{ gridColumn: 2, gridRow: `1 / span ${rows.length}` }} aria-hidden>
        {ticks(lo, hi).map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={0} y2={rows.length * rowH} stroke="var(--hairline)" />
            <text x={x(t)} y={rows.length * rowH + 16} textAnchor="middle" fontSize="10" fill="var(--fg-subtle)" fontFamily="var(--font-mono)">{fmt(t)}</text>
          </g>
        ))}
        <line x1={x(0)} x2={x(0)} y1={0} y2={rows.length * rowH} stroke="var(--fg-subtle)" strokeDasharray="4 3" />
        {rows.map((r, i) => {
          const cy = i * rowH + rowH / 2 - 3;
          return (
            <g key={r.key} className="animate-enter" style={{ animationDelay: `${i * 40}ms` }}>
              <line x1={x(r.lo)} x2={x(r.hi)} y1={cy} y2={cy} stroke={r.color} strokeWidth={5} strokeLinecap="round" opacity={0.9} />
              {r.dateLo != null && r.dateHi != null && (
                <line x1={x(r.dateLo)} x2={x(r.dateHi)} y1={cy + 8} y2={cy + 8} stroke={r.color} strokeWidth={1.5} opacity={0.7} />
              )}
              <circle cx={x(r.mean)} cy={cy} r={5.5} fill={r.color} stroke="var(--surface-1)" strokeWidth={2} />
            </g>
          );
        })}
      </svg>
      <p id={`${id}-cap`} className="col-span-3 mt-7 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fg-subtle">
        <span>{axisLabel}</span>
        <span className="flex items-center gap-1.5"><span aria-hidden className="h-[5px] w-4 rounded-full bg-[var(--fg-subtle)]" />95% CI, session clusters</span>
        <span className="flex items-center gap-1.5"><span aria-hidden className="h-px w-4 bg-[var(--fg-subtle)]" />95% CI, date clusters</span>
        <span className="flex items-center gap-1.5"><span aria-hidden className="h-3 w-0 border-l border-dashed border-[var(--fg-subtle)]" />{reference}</span>
      </p>
    </div>
  );
}
