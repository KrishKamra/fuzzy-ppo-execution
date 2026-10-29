import { useMemo } from "react";

import type { OrderTrace } from "@/api/client";
import { EChart } from "@/components/charts/EChart";
import { useInk } from "@/lib/ink";
import { useReplay } from "@/store/replay";
import { useUi } from "@/store/ui";

import { ChartLegend } from "./ChartLegend";
import { costRaceOption, GROUP, inventoryOption, priceOption } from "./charts";
import { fills, standings } from "./model";

/** The three linked charts on one minute axis: price and fills, inventory left, and the cost race vs VWAP. */
export function ExecuteCharts({ trace }: { trace: OrderTrace }) {
  const k = useReplay((s) => s.k);
  const focusName = useReplay((s) => s.focus);
  const setFocus = useReplay((s) => s.setFocus);
  const theme = useUi((s) => s.effectiveTheme());
  const ink = useInk(theme);
  const focus = trace.policies.find((p) => p.name === focusName);

  const price = useMemo(() => priceOption(trace, k, focus, ink), [trace, k, focus, ink]);
  const inventory = useMemo(() => inventoryOption(trace, k, focusName, ink), [trace, k, focusName, ink]);
  const race = useMemo(() => costRaceOption(trace, k, focusName, ink), [trace, k, focusName, ink]);

  // Text equivalents for screen readers, refreshed as the replay moves.
  const { order } = trace;
  const nFills = focus ? fills(trace, focus, k).length : 0;
  const leader = standings(trace, k)[0];
  const stock = order.security.replace(" IS Equity", "");
  const onSeriesClick = (id: string) => trace.policies.some((p) => p.name === id && p.name !== "vwap") && setFocus(id);

  return (
    <div className="space-y-1">
      <ChartLegend trace={trace} />
      <EChart group={GROUP} option={price} className="h-[260px]"
        label={`${stock} mid price ${order.date} from ${trace.market.time[0]}, order ${order.start}–${order.end}; ${nFills} fills by ${focus?.label ?? "the focus policy"} so far`} />
      <EChart group={GROUP} option={inventory} className="h-[150px]" onSeriesClick={onSeriesClick}
        label="Share of the order still to execute, per policy" />
      <EChart group={GROUP} option={race} className="h-[190px]" onSeriesClick={onSeriesClick}
        label={`Running cost minus VWAP in basis points; ${leader ? `${leader.label} leads` : ""}`} />
    </div>
  );
}
