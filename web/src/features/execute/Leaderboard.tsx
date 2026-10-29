import { clsx } from "clsx";
import { Check } from "lucide-react";

import type { OrderTrace } from "@/api/client";
import { seriesColor } from "@/lib/palette";
import { useReplay } from "@/store/replay";
import { useUi } from "@/store/ui";

import { REFERENCE, splitLabel, standings } from "./model";

const signed = (x: number) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(2)}`;

/**
 * Live standings: cost so far against VWAP on the same order. It is also the table view of the charts (the
 * light-mode relief rule), so every number on the charts is readable here as text. Click a row to focus it.
 */
export function Leaderboard({ trace }: { trace: OrderTrace }) {
  const k = useReplay((s) => s.k);
  const focus = useReplay((s) => s.focus);
  const setFocus = useReplay((s) => s.setFocus);
  const theme = useUi((s) => s.effectiveTheme());
  const rows = standings(trace, k);
  return (
    <table className="w-full table-fixed text-sm">
      <caption className="sr-only">Cost so far by policy, against VWAP on the same order</caption>
      <colgroup>
        <col />
        <col className="w-[4.75rem]" />
        <col className="w-[4rem]" />
      </colgroup>
      <thead>
        <tr className="text-left text-xs text-fg-subtle">
          <th scope="col" className="pb-2 font-medium">Policy</th>
          <th scope="col" className="pb-2 text-right font-medium" title="Running cost minus VWAP's, in basis points of the order's value">vs VWAP</th>
          <th scope="col" className="pb-2 text-right font-medium" title="Running cost (impact, spread, fees, risk), bps">Cost</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const ref = r.name === REFERENCE;
          const focused = focus === r.name;
          const [name, qualifier] = splitLabel(r.label);
          const detail = [qualifier, `${Math.round(r.done * 100)}% done`, r.passiveShare > 0 ? `${Math.round(r.passiveShare * 100)}% passive` : null]
            .filter(Boolean)
            .join(" · ");
          return (
            <tr
              key={r.name}
              aria-selected={focused}
              onClick={() => !ref && setFocus(r.name)}
              className={clsx("border-t border-hairline align-top", !ref && "cursor-pointer hover:bg-surface-2", focused && "bg-surface-2")}
            >
              <th scope="row" className="py-2 pr-2 text-left font-normal">
                <span className="flex min-w-0 items-center gap-2">
                  <span aria-hidden className="size-2 shrink-0 rounded-full" style={{ background: seriesColor(r.name, theme) }} />
                  <span className={clsx("truncate", ref ? "text-fg-muted" : "text-fg")} title={r.label}>{name}</span>
                  {r.finished && <Check aria-label="finished" className="size-3 shrink-0 text-fg-subtle" />}
                </span>
                <span className="tabular mt-0.5 block pl-4 text-[0.7rem] leading-snug text-fg-subtle">{detail}</span>
              </th>
              <td className={clsx("tabular py-2 text-right font-mono", ref ? "text-fg-subtle" : r.vsReference !== null && r.vsReference < 0 ? "text-fg" : "text-fg-muted")}>
                {ref ? "ref" : r.vsReference === null ? "–" : signed(r.vsReference)}
              </td>
              <td className="tabular py-2 text-right font-mono text-fg-muted">{r.costSoFar.toFixed(2)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
