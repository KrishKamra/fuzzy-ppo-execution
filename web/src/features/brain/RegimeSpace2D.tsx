import type { Rules } from "@/api/client";
import type { Theme } from "@/lib/palette";

import { INPUT_NAMES, paceColor } from "./fuzzy";

const LIM = 3.2;
const S = 200;
const p = (v: number) => ((Math.max(-LIM, Math.min(LIM, v)) + LIM) / (2 * LIM)) * S;

/** Without WebGL: the same space as two projections sharing the volatility axis (x = an input, y = volatility). */
export function RegimeSpace2D({ rules, path, k, active, hover, onHover, theme }: {
  rules: Rules; path: number[][]; k: number; active: number | null; hover: number | null; onHover: (r: number | null) => void; theme: Theme;
}) { // fmt: skip
  const shown = path.slice(0, Math.max(1, k));
  const cur = shown[shown.length - 1];
  return (
    <div className="grid h-full grid-cols-1 gap-4 p-2 sm:grid-cols-2" aria-label="2D projections of the regime space">
      {[0, 1].map((xi) => (
        <figure key={xi} className="flex flex-col">
          <svg viewBox={`-28 -8 ${S + 36} ${S + 36}`} className="w-full" role="img"
            aria-label={`${INPUT_NAMES[xi]} against volatility: rules as ellipses and the order's path`}>
            <rect x={0} y={0} width={S} height={S} fill="none" stroke="var(--hairline-strong)" />
            {rules.terciles[xi]?.map((c) => <line key={c} x1={p(c)} x2={p(c)} y1={0} y2={S} stroke="var(--hairline)" strokeDasharray="3 3" />)}
            {rules.terciles[2]?.map((c) => <line key={c} y1={S - p(c)} y2={S - p(c)} x1={0} x2={S} stroke="var(--hairline)" strokeDasharray="3 3" />)}
            {rules.rules.map((r, i) => (
              <ellipse key={r.rule} cx={p(r.centres[xi] ?? 0)} cy={S - p(r.centres[2] ?? 0)}
                rx={((r.widths[xi] ?? 1) / (2 * LIM)) * S} ry={((r.widths[2] ?? 1) / (2 * LIM)) * S}
                fill={paceColor(r.pace, theme)} fillOpacity={i === active ? 0.42 : i === hover ? 0.32 : 0.14}
                stroke={paceColor(r.pace, theme)} strokeOpacity={0.6}
                onMouseEnter={() => onHover(i)} onMouseLeave={() => onHover(null)}
                style={{ transition: "fill-opacity 200ms ease" }} />
            ))}
            <polyline points={shown.map((x) => `${p(x[xi] ?? 0)},${S - p(x[2] ?? 0)}`).join(" ")} fill="none" stroke="var(--accent-2)" strokeWidth={1.5} />
            {cur && <circle cx={p(cur[xi] ?? 0)} cy={S - p(cur[2] ?? 0)} r={4} fill="var(--fg)" />}
            <text x={S / 2} y={S + 22} textAnchor="middle" fontSize="10" fill="var(--fg-subtle)">{INPUT_NAMES[xi]} →</text>
            <text x={-16} y={S / 2} textAnchor="middle" fontSize="10" fill="var(--fg-subtle)" transform={`rotate(-90 -16 ${S / 2})`}>{INPUT_NAMES[2]} →</text>
          </svg>
        </figure>
      ))}
    </div>
  );
}

/** WebGL is unavailable in some browsers, VMs and test environments. */
export function hasWebGL(): boolean {
  if (typeof window === "undefined" || typeof WebGLRenderingContext === "undefined") return false;
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") ?? c.getContext("webgl"));
  } catch {
    return false;
  }
}
