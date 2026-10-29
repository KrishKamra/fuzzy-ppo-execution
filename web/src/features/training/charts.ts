/**
 * Training-run views (progress.csv and events.csv, via /api/runs/{name}): the validation edge, the evolving
 * rule base and stability against the KL guard. `summarize` is pure and tested; the option builders only draw.
 */
import type { RunCurves } from "@/api/client";
import { bandSeries } from "@/components/charts/bands";
import type { ChartOption } from "@/components/charts/EChart";
import type { Ink } from "@/lib/ink";

export const KL_GUARD = 0.02; // edit_kl_max = target_kl (configs/efnn.yaml): edits above it are undone
type Num = number | null;
const col = (run: RunCurves, key: string): Num[] => (run.progress[key] as Num[] | undefined) ?? [];

export interface RunSummary {
  bestIteration: number | null;
  bestEdge: number | null; // val objective − VWAP at the best iteration (bps)
  finalRules: number | null;
  edits: Record<"add" | "merge" | "prune", number>;
  rejected: number;
  maxEditKl: number | null;
}

export function summarize(run: RunCurves): RunSummary {
  const it = col(run, "iteration");
  const val = col(run, "val_vs_vwap");
  let best: number | null = null;
  val.forEach((v, i) => {
    if (v !== null && (best === null || v < (val[best] ?? Infinity))) best = i;
  });
  const rules = col(run, "n_rules").filter((x): x is number => x !== null);
  const edits = { add: 0, merge: 0, prune: 0 };
  let rejected = 0;
  let maxKl: number | null = null;
  for (const e of run.events) {
    const kind = String(e.kind) as keyof typeof edits;
    if (Number(e.accepted) === 1) {
      if (kind in edits) edits[kind] += 1;
      const kl = Number(e.kl);
      maxKl = maxKl === null ? kl : Math.max(maxKl, kl);
    } else rejected += 1;
  }
  return {
    bestIteration: best === null ? null : (it[best] ?? null),
    bestEdge: best === null ? null : (val[best] ?? null),
    finalRules: rules.length ? (rules[rules.length - 1] ?? null) : null,
    edits,
    rejected,
    maxEditKl: maxKl,
  };
}

function base(ink: Ink, yName: string) {
  return {
    animationDuration: 500,
    textStyle: { fontFamily: "Inter Variable, sans-serif", color: ink.muted },
    grid: { left: 56, right: 16, top: 36, bottom: 32 },
    tooltip: { trigger: "axis", backgroundColor: ink.surface2, borderColor: ink.hairline, textStyle: { color: ink.fg, fontSize: 12 } },
    xAxis: { type: "value", min: 0, max: "dataMax", name: "iteration", nameLocation: "end", nameTextStyle: { color: ink.subtle }, axisLabel: { color: ink.subtle }, splitLine: { show: false }, axisLine: { lineStyle: { color: ink.hairline } } },
    yAxis: { type: "value", name: yName, nameTextStyle: { color: ink.subtle }, axisLabel: { color: ink.subtle }, splitLine: { lineStyle: { color: ink.hairline } } },
  }; // fmt: skip
}

const pairs = (x: Num[], y: Num[]) => x.map((v, i) => [v, y[i] ?? null]).filter(([a, b]) => a !== null && b !== null);

export function edgeOption(run: RunCurves, ink: Ink, color: string): ChartOption {
  const it = col(run, "iteration");
  const val = col(run, "val_vs_vwap");
  const se = col(run, "val_vs_vwap_se");
  const band = it.flatMap((x, i) => {
    const v = val[i];
    return x === null || v === null || v === undefined ? [] : [{ x, lo: v - 1.96 * (se[i] ?? 0), hi: v + 1.96 * (se[i] ?? 0) }];
  });
  const s = summarize(run);
  return {
    ...base(ink, "bps vs VWAP"),
    series: [
      bandSeries("band", band, color),
      { id: "train", name: "train", type: "line", data: pairs(it, col(run, "train_vs_vwap")), showSymbol: false, lineStyle: { color: ink.subtle, width: 1.25 }, itemStyle: { color: ink.subtle } },
      {
        id: "val", name: "validation", type: "line", data: pairs(it, val), showSymbol: false, lineStyle: { color, width: 2 }, itemStyle: { color },
        markLine: { silent: true, symbol: "none", data: [
          { yAxis: 0, lineStyle: { color: ink.subtle, type: "dashed" }, label: { show: false } },
          ...(s.bestIteration !== null ? [{ xAxis: s.bestIteration, lineStyle: { color: ink.fg, type: "dotted" }, label: { formatter: "best.pt", color: ink.muted, fontSize: 10 } }] : []),
        ] },
      },
    ],
  }; // fmt: skip
}

export function rulesOption(run: RunCurves, ink: Ink, colors: { add: string; merge: string; prune: string; line: string }): ChartOption {
  const it = col(run, "iteration");
  const accepted = run.events.filter((e) => Number(e.accepted) === 1);
  const marks = { add: "triangle", merge: "diamond", prune: "pin" } as const;
  return {
    ...base(ink, "rules"),
    legend: { top: 0, right: 0, textStyle: { color: ink.muted, fontSize: 11 }, itemWidth: 10, itemHeight: 10, data: ["add", "merge", "prune"] },
    series: [
      { id: "rules", name: "rules", type: "line", step: "end", data: pairs(it, col(run, "n_rules")), showSymbol: false, lineStyle: { color: colors.line, width: 2 }, itemStyle: { color: colors.line } },
      ...(Object.keys(marks) as (keyof typeof marks)[]).map((k) => ({
        id: k, name: k, type: "scatter", symbol: marks[k], symbolSize: 11, z: 3, itemStyle: { color: colors[k] },
        data: accepted.filter((e) => e.kind === k).map((e) => [Number(e.iteration), Number(e.n_rules)]),
      })),
    ],
  }; // fmt: skip
}

export function stabilityOption(run: RunCurves, ink: Ink, color: string): ChartOption {
  const it = col(run, "iteration");
  const edits = run.events.filter((e) => Number(e.accepted) === 1).map((e) => [Number(e.iteration), Number(e.kl)]);
  const b = base(ink, "KL");
  return {
    ...b,
    yAxis: { ...b.yAxis, min: 0, max: (v: { max: number }) => Math.max(v.max, KL_GUARD * 1.25) }, // the guard stays in view
    legend: { top: 0, right: 0, textStyle: { color: ink.muted, fontSize: 11 }, itemWidth: 12, itemHeight: 8 },
    series: [
      { id: "kl", name: "PPO update KL", type: "line", data: pairs(it, col(run, "kl")), showSymbol: false, lineStyle: { color: ink.subtle, width: 1.25 }, itemStyle: { color: ink.subtle } },
      {
        id: "edit", name: "structure-edit KL", type: "scatter", data: edits, symbolSize: 8, itemStyle: { color },
        markLine: { silent: true, symbol: "none", data: [{ yAxis: KL_GUARD, lineStyle: { color: ink.fg, type: "dashed" }, label: { formatter: "KL guard 0.02", position: "insideEndTop", color: ink.muted, fontSize: 10 } }] },
      },
    ],
  }; // fmt: skip
}
