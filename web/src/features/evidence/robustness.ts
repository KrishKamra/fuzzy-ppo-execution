import { bandSeries } from "@/components/charts/bands";
import type { ChartOption } from "@/components/charts/EChart";
import type { Ink } from "@/lib/ink";
import { seriesColor } from "@/lib/palette";

import { LABELS, type Row } from "./data";

export const SWEEPS = {
  gamma: { id: "robust_gamma", param: "gamma", label: "Transient impact strength γ", train: [0.1, 0.5] as const },
  tau: { id: "robust_tau", param: "tau_res_min", label: "Impact resilience τ (minutes)", train: [2, 15] as const },
  eta: { id: "robust_eta", param: "eta", label: "Temporary impact η", train: [0.25, 1] as const },
};
export type SweepKey = keyof typeof SWEEPS;

/** Ensembles vs VWAP across one impact parameter: lines with 95% CI bands, the training range shaded. */
export function robustnessOption(rows: Row[], sweep: SweepKey, ink: Ink): ChartOption {
  const s = SWEEPS[sweep];
  const series = (["mlp", "efnn"] as const).flatMap((p) => {
    const pts = rows
      .filter((r) => r.policy === p && r.reference === "vwap")
      .map((r) => ({ x: Number(r[s.param]), mean: Number(r.mean), lo: Number(r.lo), hi: Number(r.hi) }))
      .sort((a, b) => a.x - b.x);
    const color = seriesColor(p, ink.theme);
    return [
      bandSeries(`${p}-band`, pts, color),
      { id: p, name: LABELS[p], type: "line", data: pts.map((q) => [q.x, q.mean]), symbolSize: 7, lineStyle: { color, width: 2 }, itemStyle: { color }, z: 3 },
    ];
  }); // fmt: skip
  return {
    animationDuration: 500,
    textStyle: { fontFamily: "Inter Variable, sans-serif", color: ink.muted },
    grid: { left: 52, right: 16, top: 16, bottom: 44 },
    tooltip: {
      trigger: "axis", backgroundColor: ink.surface2, borderColor: ink.hairline, textStyle: { color: ink.fg, fontSize: 12 },
      valueFormatter: (v: unknown) => (typeof v === "number" ? `${v.toFixed(3)} bps` : "–"),
    },
    xAxis: {
      type: "value", min: "dataMin", max: "dataMax", name: s.label, nameLocation: "middle", nameGap: 28, nameTextStyle: { color: ink.subtle },
      axisLabel: { color: ink.subtle }, splitLine: { show: false }, axisLine: { lineStyle: { color: ink.hairline } },
    },
    yAxis: { type: "value", name: "bps vs VWAP", nameTextStyle: { color: ink.subtle }, axisLabel: { color: ink.subtle }, splitLine: { lineStyle: { color: ink.hairline } } },
    series: [
      ...series,
      {
        id: "marks", type: "line", data: [], silent: true,
        markArea: { silent: true, itemStyle: { color: ink.hairline, opacity: 0.6 }, label: { show: true, position: "insideTop", color: ink.subtle, fontSize: 10, formatter: "training range" }, data: [[{ xAxis: s.train[0] }, { xAxis: s.train[1] }]] },
        markLine: { silent: true, symbol: "none", lineStyle: { color: ink.subtle, type: "dashed" }, label: { show: false }, data: [{ yAxis: 0 }] },
      },
    ],
  }; // fmt: skip
}
