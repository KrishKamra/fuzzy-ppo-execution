/**
 * ECharts options for the Execute screen. Mark specs follow the dataviz skill: 2 px lines, ≥ 8 px markers,
 * recessive grid and axes, text in ink tokens (never series colours) and a crosshair tooltip. Series identity
 * comes from the HTML legend above the charts and the standings table (the table view that satisfies the
 * light-mode relief rule); end labels were dropped because every line ends at the same minute and collides.
 * The replay drives every update, so ECharts' own animation is off.
 */
import type { OrderTrace, PolicyTrace } from "@/api/client";
import type { ChartOption } from "@/components/charts/EChart";
import type { Ink } from "@/lib/ink";
import { seriesColor, seriesDashed } from "@/lib/palette";

import { costRaceSeries, cursorIndex, fills, inventorySeries, marketSeries, REFERENCE, startIndex } from "./model";

const FONT = "IBM Plex Sans, ui-sans-serif, system-ui, sans-serif";
const MONO = "IBM Plex Mono, ui-monospace, monospace";
export const GROUP = "execute";

function base(trace: OrderTrace, ink: Ink, opts: { top: number; bottom: number; showTime: boolean }) {
  return {
    animation: false,
    textStyle: { fontFamily: FONT, color: ink.muted },
    grid: { left: 56, right: 20, top: opts.top, bottom: opts.bottom },
    xAxis: {
      type: "category",
      data: trace.market.time,
      boundaryGap: false,
      axisLine: { lineStyle: { color: ink.hairline } },
      axisTick: { show: false },
      axisLabel: { show: opts.showTime, color: ink.subtle, fontSize: 11, interval: 14, hideOverlap: true },
      axisPointer: { lineStyle: { color: ink.muted, width: 1, type: "dashed" } },
    },
    tooltip: {
      trigger: "axis",
      backgroundColor: ink.surface2,
      borderColor: ink.hairline,
      borderWidth: 1,
      padding: [8, 10],
      textStyle: { color: ink.fg, fontFamily: FONT, fontSize: 12 },
      extraCssText: "border-radius:8px; box-shadow:0 12px 32px -12px rgba(0,0,0,.5);",
      axisPointer: { type: "line" },
      valueFormatter: (v: unknown) => (typeof v === "number" ? v.toFixed(2) : "–"),
    },
  };
}

function yAxis(ink: Ink, name: string, extra: Record<string, unknown> = {}) {
  return {
    type: "value",
    name,
    nameTextStyle: { color: ink.subtle, fontSize: 11, align: "left", padding: [0, 0, 0, -40] },
    axisLabel: { color: ink.subtle, fontSize: 11, fontFamily: MONO },
    splitLine: { lineStyle: { color: ink.hairline } },
    ...extra,
  };
}

/** Vertical markers: where the order starts, and the replay cursor. */
function markers(trace: OrderTrace, k: number, ink: Ink, label: boolean) {
  return {
    silent: true,
    symbol: "none",
    animation: false,
    data: [
      { xAxis: startIndex(trace), lineStyle: { color: ink.subtle, type: "dotted", width: 1 }, label: { show: label, formatter: "start", color: ink.subtle, fontSize: 10 } },
      { xAxis: cursorIndex(trace, k), lineStyle: { color: ink.accent, width: 1.5 }, label: { show: false } },
    ],
  }; // fmt: skip
}

function line(p: PolicyTrace, data: (number | null)[], ink: Ink, unit: string, focus: boolean) {
  const color = seriesColor(p.name, ink.theme);
  return {
    id: p.name,
    name: p.label,
    type: "line",
    data,
    showSymbol: false,
    connectNulls: false,
    step: false,
    z: p.name === REFERENCE ? 1 : focus ? 4 : 3,
    lineStyle: { color, width: focus ? 2.5 : 2, type: seriesDashed(p.name) || p.name === REFERENCE ? [6, 4] : "solid" },
    itemStyle: { color },
    emphasis: { focus: "series" },
    tooltip: { valueFormatter: (v: unknown) => (typeof v === "number" ? `${v.toFixed(unit === "%" ? 0 : 2)}${unit}` : "–") },
  };
}

export function priceOption(trace: OrderTrace, k: number, focus: PolicyTrace | undefined, ink: Ink): ChartOption {
  const lo = marketSeries(trace, k, "bid");
  const hi = marketSeries(trace, k, "ask");
  const band = hi.map((h, i) => (h === null || lo[i] === null ? null : h - (lo[i] as number)));
  const pts = focus ? fills(trace, focus, k) : [];
  const qMax = Math.max(1, ...pts.map((f) => f.q));
  const color = focus ? seriesColor(focus.name, ink.theme) : ink.accent;
  const opt = base(trace, ink, { top: 28, bottom: 12, showTime: false });
  return {
    ...opt,
    yAxis: yAxis(ink, "₹", { scale: true, splitNumber: 4 }),
    series: [
      // Bid-ask band: an invisible bid line, and the spread stacked on it as a soft area.
      { id: "bid", name: "Bid", type: "line", data: lo, stack: "band", showSymbol: false, lineStyle: { opacity: 0 }, silent: true, z: 1 },
      { id: "spread", name: "Spread", type: "line", data: band, stack: "band", showSymbol: false, lineStyle: { opacity: 0 }, areaStyle: { color: ink.hairline, opacity: 1 }, silent: true, tooltip: { show: false }, z: 1 },
      {
        id: "mid", name: "Mid", type: "line", data: marketSeries(trace, k, "mid"), showSymbol: false,
        lineStyle: { color: ink.muted, width: 1.5 }, itemStyle: { color: ink.muted }, z: 2,
        markLine: markers(trace, k, ink, true),
        // The 30 minutes before the order are context: dimmed.
        markArea: { silent: true, itemStyle: { color: ink.hairline, opacity: 0.5 }, data: [[{ xAxis: 0 }, { xAxis: startIndex(trace) }]] },
      },
      {
        id: "fills", name: focus ? `${focus.label} fills` : "Fills", type: "scatter", z: 5,
        data: pts.map((f) => ({
          value: [f.index, f.price],
          symbolSize: 6 + 8 * Math.sqrt(f.q / qMax),
          // Mostly passive (limit) fills are hollow rings; aggressive fills are solid.
          itemStyle: f.passive >= f.q / 2
            ? { color: "transparent", borderColor: color, borderWidth: 1.5 }
            : { color, borderColor: ink.surface, borderWidth: 1.5 },
          q: f.q, passive: f.passive,
        })),
        tooltip: { trigger: "item", formatter: (x: { data: { value: [number, number]; q: number; passive: number } }) =>
          `${Math.round(x.data.q).toLocaleString()} shares at ₹${x.data.value[1].toFixed(2)}${x.data.passive > 0 ? ` · ${Math.round(x.data.passive).toLocaleString()} passive` : ""}` },
      },
    ],
  }; // fmt: skip
}

export function inventoryOption(trace: OrderTrace, k: number, focus: string | null, ink: Ink): ChartOption {
  return {
    ...base(trace, ink, { top: 28, bottom: 12, showTime: false }),
    yAxis: yAxis(ink, "% left", { min: 0, max: 100, splitNumber: 2 }),
    series: trace.policies.map((p, i) => ({
      ...line(p, inventorySeries(trace, p, k), ink, "%", p.name === focus),
      ...(i === 0 ? { markLine: markers(trace, k, ink, false) } : {}),
    })),
  };
}

export function costRaceOption(trace: OrderTrace, k: number, focus: string | null, ink: Ink): ChartOption {
  return {
    ...base(trace, ink, { top: 28, bottom: 28, showTime: true }),
    yAxis: yAxis(ink, "bps vs VWAP"),
    series: trace.policies.map((p, i) => ({
      ...line(p, costRaceSeries(trace, p, k), ink, "", p.name === focus),
      ...(i === 0
        ? {
            markLine: {
              ...markers(trace, k, ink, false),
              data: [...markers(trace, k, ink, false).data, { yAxis: 0, lineStyle: { color: ink.subtle, type: "solid", width: 1 }, label: { show: false } }],
            },
          }
        : {}),
    })),
  }; // fmt: skip
}
