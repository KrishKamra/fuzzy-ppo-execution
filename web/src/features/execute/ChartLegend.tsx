import { clsx } from "clsx";

import type { OrderTrace } from "@/api/client";
import { seriesColor, seriesDashed } from "@/lib/palette";
import { useReplay } from "@/store/replay";
import { useUi } from "@/store/ui";

import { REFERENCE } from "./model";

/** One legend for the three linked charts: a line swatch per policy (dashed where the chart draws it dashed).
 *  Clicking a policy makes it the focus (its fills on the price chart, its decisions in the rail). */
export function ChartLegend({ trace }: { trace: OrderTrace }) {
  const theme = useUi((s) => s.effectiveTheme());
  const focus = useReplay((s) => s.focus);
  const setFocus = useReplay((s) => s.setFocus);
  return (
    <ul aria-label="Legend" className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-2 pb-1 text-xs">
      {trace.policies.map((p) => {
        const ref = p.name === REFERENCE;
        const dashed = ref || seriesDashed(p.name);
        const color = seriesColor(p.name, theme);
        return (
          <li key={p.name}>
            <button
              type="button"
              disabled={ref}
              aria-pressed={focus === p.name}
              onClick={() => setFocus(p.name)}
              className={clsx("flex items-center gap-2 rounded px-1 py-0.5", focus === p.name ? "text-fg" : "text-fg-muted hover:text-fg", ref && "cursor-default")}
            >
              <svg aria-hidden width="18" height="6" className="shrink-0">
                <line x1="1" y1="3" x2="17" y2="3" stroke={color} strokeWidth={focus === p.name ? 3 : 2} strokeDasharray={dashed ? "4 3" : undefined} strokeLinecap="round" />
              </svg>
              {p.label}
              {ref && <span className="text-fg-subtle">(reference)</span>}
            </button>
          </li>
        );
      })}
      <li className="ml-auto flex items-center gap-3 text-fg-subtle">
        <span className="flex items-center gap-1.5"><span aria-hidden className="size-2 rounded-full bg-[var(--fg-muted)]" />aggressive fill</span>
        <span className="flex items-center gap-1.5"><span aria-hidden className="size-2 rounded-full border-[1.5px] border-[var(--fg-muted)]" />passive fill</span>
      </li>
    </ul>
  );
}
