import { clsx } from "clsx";
import { AlertTriangle, CheckCircle2, CircleDashed, CircleSlash } from "lucide-react";
import { type ReactNode, useMemo, useState } from "react";

import { useResults } from "@/api/queries";
import { EChart } from "@/components/charts/EChart";
import { Eyebrow, PageHeader, Panel, Segmented } from "@/components/ui/primitives";
import { CLAIMS, ENSEMBLES, LABELS, num, REPLICATION_VS_MLP, REPLICATION_VS_VWAP, row, type Status, table } from "@/features/evidence/data";
import { type ForestRow, ForestPlot } from "@/features/evidence/ForestPlot";
import { robustnessOption, type SweepKey, SWEEPS } from "@/features/evidence/robustness";
import { useInk } from "@/lib/ink";
import { seriesColor } from "@/lib/palette";
import { useUi } from "@/store/ui";

const STATUS: Record<Status, { label: string; icon: ReactNode; cls: string }> = {
  supported: { label: "Supported", icon: <CheckCircle2 aria-hidden className="size-4" />, cls: "text-good" },
  partly: { label: "Partly", icon: <CircleDashed aria-hidden className="size-4" />, cls: "text-accent-2" },
  not: { label: "Not established", icon: <CircleSlash aria-hidden className="size-4" />, cls: "text-fg-subtle" },
};

const p = (x: number | null) => (x === null ? "" : x < 0.001 ? "Holm p<0.001" : `Holm p=${x.toFixed(3)}`);

function Section({ id, title, takeaway, source, children }: { id: string; title: string; takeaway: string; source: string; children: ReactNode }) {
  return (
    <Panel id={id} className="scroll-mt-20 p-6">
      <Eyebrow>{title}</Eyebrow>
      <h2 className="mt-2 text-lg font-semibold tracking-tight text-balance">{takeaway}</h2>
      <div className="mt-5">{children}</div>
      <p className="mt-4 text-xs text-fg-subtle">Source: {source}</p>
    </Panel>
  );
}

export function Evidence() {
  const results = useResults();
  const theme = useUi((s) => s.effectiveTheme());
  const ink = useInk(theme);
  const [run, setRun] = useState<"ours" | "replication" | "both">("both");
  const [sweep, setSweep] = useState<SweepKey>("gamma");
  const r = results.data;
  const color = (k: string) => seriesColor(k, theme);

  const vsVwap = useMemo<ForestRow[]>(() => {
    const t = table(r, "test_vs_vwap");
    return ENSEMBLES.flatMap((e) => {
      const ours = row(t, e);
      const rep = REPLICATION_VS_VWAP[e]!;
      const out: ForestRow[] = [];
      if (run !== "replication" && ours)
        out.push({ key: `${e}-ours`, label: LABELS[e]!, sublabel: "ours · 5 seeds", mean: num(ours, "mean") ?? 0, lo: num(ours, "lo") ?? 0, hi: num(ours, "hi") ?? 0, dateLo: num(ours, "date_lo"), dateHi: num(ours, "date_hi"), color: color(e), note: p(num(ours, "holm_p")) });
      if (run !== "ours")
        out.push({ key: `${e}-rep`, label: LABELS[e]!, sublabel: "replication · 3 seeds", mean: rep.mean, lo: rep.lo, hi: rep.hi, dateLo: rep.dateLo, dateHi: rep.dateHi, color: color(e), note: p(rep.holm) });
      return out;
    });
  }, [r, run, theme]);

  const architectures = useMemo<ForestRow[]>(() => {
    const t = table(r, "test_vs_mlp");
    return (["anfis", "efnn"] as const).flatMap((e) => {
      const ours = row(t, e, "mlp");
      const rep = REPLICATION_VS_MLP[e]!;
      return [
        ...(ours ? [{ key: `${e}-o`, label: `${LABELS[e]} − MLP`, sublabel: "ours", mean: num(ours, "mean") ?? 0, lo: num(ours, "lo") ?? 0, hi: num(ours, "hi") ?? 0, dateLo: num(ours, "date_lo"), dateHi: num(ours, "date_hi"), color: color(e), note: p(num(ours, "holm_p")) }] : []),
        { key: `${e}-r`, label: `${LABELS[e]} − MLP`, sublabel: "replication", mean: rep.mean, lo: rep.lo, hi: rep.hi, color: color(e), note: p(rep.holm) },
      ];
    });
  }, [r, theme]);

  const fills = useMemo<ForestRow[]>(() => {
    const bar = table(r, "bar_vs_vwap");
    const tick = table(r, "tick_vs_vwap");
    return ENSEMBLES.flatMap((e) =>
      ([["BarSim", bar], ["TickSim", tick]] as const).flatMap(([sim, t]) => {
        const x = row(t, e);
        return x ? [{ key: `${e}-${sim}`, label: LABELS[e]!, sublabel: sim, mean: num(x, "mean") ?? 0, lo: num(x, "lo") ?? 0, hi: num(x, "hi") ?? 0, dateLo: num(x, "date_lo"), dateHi: num(x, "date_hi"), color: color(e) }] : [];
      }),
    );
  }, [r, theme]);

  const limit = useMemo<ForestRow[]>(() => {
    const v = table(r, "limit_vs_vwap");
    const pas = table(r, "limit_vs_passive");
    const mk = (t: typeof v, k: string, ref: string, colorKey: string) => {
      const x = row(t, k, ref);
      return x ? [{ key: `${k}-${ref}`, label: LABELS[k] ?? k, sublabel: `vs ${ref === "vwap" ? "VWAP" : "always passive"}`, mean: num(x, "mean") ?? 0, lo: num(x, "lo") ?? 0, hi: num(x, "hi") ?? 0, dateLo: num(x, "date_lo"), dateHi: num(x, "date_hi"), color: color(colorKey), note: ref === "vwap" ? "" : p(num(x, "holm_p")) }] : [];
    };
    return [...mk(v, "efnn", "vwap", "efnn"), ...mk(v, "vwap_passive100", "vwap", "vwap_passive100"), ...mk(v, "lip_efnn", "vwap", "efnn_limit"), ...mk(v, "lip_mlp", "vwap", "mlp_limit"),
            ...mk(pas, "lip_efnn", "vwap_passive100", "efnn_limit"), ...mk(pas, "lip_mlp", "vwap_passive100", "mlp_limit")]; // fmt: skip
  }, [r, theme]);

  const robust = useMemo(() => robustnessOption(table(r, SWEEPS[sweep].id), sweep, ink), [r, sweep, ink]);
  const regimes = table(r, "test_by_regime").filter((x) => ENSEMBLES.includes(x.policy as (typeof ENSEMBLES)[number]));

  if (results.error)
    return <p role="alert" className="flex items-center gap-2 text-sm text-fg-muted"><AlertTriangle aria-hidden className="size-4 text-bad" /> {results.error.message}</p>;

  return (
    <div className="space-y-5">
      <PageHeader eyebrow="Evidence" title="What the statistics support, and what they do not" />
      <Panel className="p-6">
        <Eyebrow>Claims</Eyebrow>
        <ul className="mt-4 grid gap-3 md:grid-cols-2">
          {CLAIMS.map((c) => {
            const s = STATUS[c.status];
            return (
              <li key={c.claim}>
                <a href={`#${c.anchor}`} className="gradient-border block h-full rounded-[var(--radius-control)] border border-hairline bg-surface-2 p-4 transition-colors hover:bg-surface-3">
                  <span className={clsx("flex items-center gap-1.5 text-xs font-medium", s.cls)}>{s.icon}{s.label}</span>
                  <span className="mt-1.5 block font-medium">{c.claim}</span>
                  <span className="mt-1 block text-sm text-fg-muted">{c.evidence}</span>
                </a>
              </li>
            );
          })}
        </ul>
        <p className="mt-4 text-xs text-fg-subtle">
          All costs in basis points of the order's value, in simulation (Bloomberg data, three Nifty-50 stocks); paired episodes; 95% session-cluster bootstrap CIs; Holm-adjusted Wilcoxon tests. Not a claim about live trading.
        </p>
      </Panel>

      {!r ? (
        <div className="space-y-5" aria-busy="true"><div className="skeleton h-72" /><div className="skeleton h-56" /></div>
      ) : (
        <>
          <Section id="vs-vwap" title="Out of sample" takeaway="Every ensemble beats VWAP on the held-out test split, in two independent runs" source="runs/reports/final_test_vs_vwap.csv; replication: docs/RESULTS.md">
            <div className="mb-4">
              <Segmented label="Run" value={run} onChange={setRun} options={[{ value: "ours", label: "Our run" }, { value: "replication", label: "Replication" }, { value: "both", label: "Both" }]} />
            </div>
            <ForestPlot key={run} rows={vsVwap} axisLabel="objective − VWAP, test split (bps; below zero is cheaper)" />
          </Section>

          <Section id="architectures" title="Architectures" takeaway="Fuzzy vs neural network: no difference survives the multiple-comparison correction" source="runs/reports/final_test_vs_group.csv; replication: docs/RESULTS.md">
            <ForestPlot rows={architectures} reference="PPO-MLP" axisLabel="difference to PPO-MLP, test split (bps)" />
          </Section>

          <Section id="fills" title="Realistic fills" takeaway="Replayed against the real order book, the edge holds and grows" source="runs/reports/m5_val_{bar,tick}_vs_vwap.csv (24 val tick sessions, identical episodes)">
            <ForestPlot rows={fills} axisLabel="objective − VWAP, BarSim vs TickSim (bps)" />
          </Section>

          <Section id="limit" title="Limit orders" takeaway="Learned limit orders beat VWAP and the naive always-passive rule" source="runs/reports/m5_lip_val_ticksim_vs_{vwap,group}.csv. Passive fills are counterfactual: absolute gains are an upper bound; the date Holm p of 0.016 is the floor for 8 dates.">
            <ForestPlot rows={limit} axisLabel="cost difference, TickSim (bps)" />
          </Section>

          <Section id="robustness" title="Robustness" takeaway="The edge is transient-impact management: it vanishes without it and grows with it" source="runs/reports/robust_{gamma,tau_res_min,eta}.csv (validation, 5-seed ensembles)">
            <Segmented<SweepKey> label="Impact parameter" value={sweep} onChange={setSweep}
              options={(Object.keys(SWEEPS) as SweepKey[]).map((k) => ({ value: k, label: SWEEPS[k].label.split(" (")[0]! }))} />
            <EChart option={robust} className="mt-3 h-[300px]" label={`${SWEEPS[sweep].label}: PPO-MLP and PPO-eFNN minus VWAP with 95% bands; the training range is shaded`} />
          </Section>

          {regimes.length > 0 && (
            <Section id="regimes" title="By regime" takeaway="Gains after both medium- and high-volatility days (the test month had no low-volatility days)" source="runs/reports/final_test_regime.csv">
              <table className="w-full max-w-xl text-sm">
                <thead><tr className="text-left text-xs text-fg-subtle"><th className="pb-2 font-medium">Previous-day volatility</th>{ENSEMBLES.map((e) => <th key={e} className="pb-2 text-right font-medium">{LABELS[e]}</th>)}</tr></thead>
                <tbody className="tabular font-mono">
                  {["low", "mid", "high"].filter((g) => regimes.some((x) => x.regime_eval_prevday === g)).map((g) => (
                    <tr key={g} className="border-t border-hairline">
                      <th scope="row" className="py-2 text-left font-sans font-normal text-fg-muted">{g}</th>
                      {ENSEMBLES.map((e) => {
                        const x = regimes.find((y) => y.policy === e && y.regime_eval_prevday === g);
                        return <td key={e} className="py-2 text-right">{num(x, "mean")?.toFixed(2) ?? "–"}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
          )}
        </>
      )}
    </div>
  );
}
