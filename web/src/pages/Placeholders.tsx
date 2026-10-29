import { AlertTriangle, Loader2 } from "lucide-react";
import type { ReactNode } from "react";

import { useMeta } from "@/api/queries";
import { PageHeader, Panel } from "@/components/ui/primitives";

/** Live facts from /api/meta, so each screen proves the wiring before its charts land (M6d-M6f). */
function MetaFacts({ pick }: { pick: (m: NonNullable<ReturnType<typeof useMeta>["data"]>) => [string, ReactNode][] }) {
  const { data, isPending, error } = useMeta();
  if (isPending)
    return (
      <p className="flex items-center gap-2 text-sm text-fg-muted">
        <Loader2 aria-hidden className="size-4 animate-spin" /> Loading from the API…
      </p>
    );
  if (error)
    return (
      <p role="alert" className="flex items-center gap-2 text-sm text-fg-muted">
        <AlertTriangle aria-hidden className="size-4 text-bad" /> {error.message}
      </p>
    );
  return (
    <dl className="grid gap-x-8 gap-y-4 sm:grid-cols-3">
      {pick(data).map(([k, v]) => (
        <div key={k}>
          <dt className="text-xs text-fg-subtle">{k}</dt>
          <dd className="tabular mt-1 font-mono text-xl">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

function Scaffold({ eyebrow, title, next, children }: { eyebrow: string; title: string; next: string; children: ReactNode }) {
  return (
    <>
      <PageHeader eyebrow={eyebrow} title={title} />
      <Panel className="p-6">
        {children}
        <p className="mt-6 border-t border-hairline pt-4 text-sm text-fg-subtle">{next}</p>
      </Panel>
    </>
  );
}

export function Brain() {
  return (
    <Scaffold eyebrow="Brain" title="The fuzzy rules and the regime space" next="The 3D regime space and rule tables arrive in M6e.">
      <MetaFacts
        pick={(m) => [
          ["Regime inputs", m.regime_inputs.length],
          ["Policies that explain themselves", m.policies.filter((p) => p.explains).length],
          ["With limit orders", m.policies.filter((p) => p.limit_orders).length],
        ]}
      />
    </Scaffold>
  );
}

export function Evidence() {
  return (
    <Scaffold eyebrow="Evidence" title="What the statistics support" next="Forest plots, the replication and the claims checklist arrive in M6f.">
      <MetaFacts pick={(m) => [["Securities", m.securities.length], ["Horizons (min)", m.horizons.join(" · ")], ["Baselines", m.policies.filter((p) => p.kind === "baseline").length]]} />
    </Scaffold>
  );
}

export function Training() {
  return (
    <Scaffold eyebrow="Training" title="How the agent and its rules learned" next="Learning curves and rule-evolution events arrive in M6f.">
      <MetaFacts pick={(m) => [["Training runs", m.runs.length], ["Window", `${m.earliest_start}–${m.latest_end}`], ["Order size (of volume)", `${m.participation[0] * 100}–${m.participation[1] * 100}%`]]} />
    </Scaffold>
  );
}

export function NotFound() {
  return <PageHeader eyebrow="404" title="Nothing here" />;
}
