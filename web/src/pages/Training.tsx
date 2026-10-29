import { AlertTriangle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { useMeta, useRun } from "@/api/queries";
import { EChart } from "@/components/charts/EChart";
import { Eyebrow, PageHeader, Panel, Segmented } from "@/components/ui/primitives";
import { edgeOption, KL_GUARD, rulesOption, stabilityOption, summarize } from "@/features/training/charts";
import { useInk } from "@/lib/ink";
import { seriesColor } from "@/lib/palette";
import { useUi } from "@/store/ui";

/** "m4_efnn_s0" → "eFNN · seed 0"; "m5_lip_efnn_s0" → "eFNN + limit orders · seed 0". */
export function runLabel(name: string): string {
  const m = /^m\d+_(lip_)?(efnn|mlp|anfis)_s(\d+)$/.exec(name);
  if (!m) return name;
  const arch = { efnn: "eFNN", mlp: "MLP", anfis: "ANFIS" }[m[2] as "efnn" | "mlp" | "anfis"];
  return `${arch}${m[1] ? " + limit orders" : ""} · seed ${m[3]}`;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="signal-ring rounded-[var(--radius-control)] border border-hairline bg-surface-2 px-4 py-3" title={hint}>
      <p className="text-xs text-fg-subtle">{label}</p>
      <p className="tabular mt-1 font-mono text-xl">{value}</p>
    </div>
  );
}

export function Training() {
  const meta = useMeta();
  const theme = useUi((s) => s.effectiveTheme());
  const ink = useInk(theme);
  const runs = meta.data?.runs ?? [];
  const [name, setName] = useState<string | null>(null);
  useEffect(() => {
    if (!name && runs[0]) setName(runs[0]);
  }, [runs, name]);
  const run = useRun(name);
  const data = run.data;
  const fuzzy = !!data && (data.progress.n_rules?.some((x) => x !== null) ?? false);
  // The rule-count line is amber, so the merge events take violet to stay visible on it.
  const accent = seriesColor(name?.includes("mlp") ? "mlp" : "efnn", theme);

  const s = useMemo(() => (data ? summarize(data) : null), [data]);
  const edge = useMemo(() => (data ? edgeOption(data, ink, accent) : null), [data, ink, accent]);
  const rules = useMemo(
    () => (data && fuzzy ? rulesOption(data, ink, { add: seriesColor("mlp", theme), merge: seriesColor("anfis", theme), prune: seriesColor("efnn_limit", theme), line: accent }) : null),
    [data, fuzzy, ink, theme, accent],
  );
  const stability = useMemo(() => (data ? stabilityOption(data, ink, accent) : null), [data, ink, accent]);

  if (meta.error || run.error)
    return <p role="alert" className="flex items-center gap-2 text-sm text-fg-muted"><AlertTriangle aria-hidden className="size-4 text-bad" /> {(meta.error ?? run.error)?.message}</p>;

  return (
    <div className="space-y-5">
      <PageHeader eyebrow="Training" title="How the agent and its rules learned">
        {runs.length > 1 && (
          <Segmented label="Training run" value={name ?? ""} onChange={setName} options={runs.map((r) => ({ value: r, label: runLabel(r) }))} />
        )}
      </PageHeader>
      {!data || !s ? (
        <div className="space-y-5" aria-busy="true"><div className="skeleton h-24" /><div className="skeleton h-72" /></div>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Selected checkpoint" value={s.bestIteration === null ? "–" : `iter ${s.bestIteration}`} hint="best.pt: the lowest validation objective" />
            <Stat label="Validation edge" value={s.bestEdge === null ? "–" : `${s.bestEdge.toFixed(2)} bps`} hint="objective − VWAP on the trainer's fixed validation episodes" />
            <Stat label="Rules at the end" value={fuzzy ? String(s.finalRules ?? "–") : "no rules (MLP)"} />
            <Stat label="Edits accepted / undone" value={fuzzy ? `${s.edits.add + s.edits.merge + s.edits.prune} / ${s.rejected}` : "–"}
              hint={`add ${s.edits.add} · merge ${s.edits.merge} · prune ${s.edits.prune}; max edit KL ${s.maxEditKl?.toFixed(4) ?? "–"} (guard ${KL_GUARD})`} />
          </div>
          <Panel className="p-6">
            <Eyebrow>Learning</Eyebrow>
            <h2 className="mt-2 text-lg font-semibold">The edge over VWAP appears within ~70 iterations and holds on validation</h2>
            {edge && <EChart option={edge} className="mt-3 h-[280px]" label="Objective minus VWAP on validation and train over PPO iterations, with the selected checkpoint" />}
          </Panel>
          <div className="grid gap-5 lg:grid-cols-2">
            <Panel className="p-6">
              <Eyebrow>Rule base</Eyebrow>
              <h2 className="mt-2 text-lg font-semibold">{fuzzy ? "Rules are added for new regimes, then merged and pruned" : "An MLP has no rule base"}</h2>
              {rules ? <EChart option={rules} className="mt-3 h-[240px]" label="Number of fuzzy rules over training, with add, merge and prune events" /> : <p className="mt-3 text-sm text-fg-muted">Pick an eFNN run to watch its rules evolve.</p>}
            </Panel>
            <Panel className="p-6">
              <Eyebrow>Stability</Eyebrow>
              <h2 className="mt-2 text-lg font-semibold">Every structural edit stays far below the KL guard</h2>
              {stability && <EChart option={stability} className="mt-3 h-[240px]" label="PPO update KL and structure-edit KL against the 0.02 guard" />}
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}
