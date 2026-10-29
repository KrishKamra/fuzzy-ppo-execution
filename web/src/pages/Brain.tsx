import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Shuffle } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";

import { api, type OrderTrace } from "@/api/client";
import { useMeta, useRules } from "@/api/queries";
import { Button, Eyebrow, PageHeader, Panel, Segmented } from "@/components/ui/primitives";
import { activeRule, paceColor, ruleSentence } from "@/features/brain/fuzzy";
import { FuzzyBrain } from "@/features/brain/FuzzyBrain";
import { hasWebGL, RegimeSpace2D } from "@/features/brain/RegimeSpace2D";
import { decisionAt } from "@/features/execute/model";
import { TransportDock } from "@/features/execute/TransportDock";
import { useInk } from "@/lib/ink";
import { startReplayLoop, useReplay } from "@/store/replay";
import { useUi } from "@/store/ui";

// three.js loads only here, and only if the browser has WebGL.
const RegimeSpace = lazy(() => import("@/features/brain/RegimeSpace"));

const prefersReducedMotion = () => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** A random validation order, run for one fuzzy policy (and VWAP): the path its rules are read along. */
function useBrainOrder(policy: string | null, seed: number) {
  return useQuery({
    queryKey: ["brain-order", policy, seed],
    queryFn: async (): Promise<OrderTrace> => {
      const t = await api.randomTicket("val", "bar", seed);
      return api.order({ ...t, policies: [policy as string, "vwap"], sim: "bar" });
    },
    enabled: !!policy,
    staleTime: Infinity,
    retry: false,
  });
}

export function Brain() {
  const meta = useMeta();
  const theme = useUi((s) => s.effectiveTheme());
  const ink = useInk(theme);
  const explainers = useMemo(() => meta.data?.policies.filter((p) => p.explains) ?? [], [meta.data]);
  const [policy, setPolicy] = useState<string | null>(null);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1e6));
  const [hover, setHover] = useState<number | null>(null);
  const webgl = useMemo(hasWebGL, []);

  useEffect(() => {
    if (!policy && explainers[0]) setPolicy(explainers[0].name);
  }, [explainers, policy]);

  const rules = useRules(policy);
  const order = useBrainOrder(policy, seed);
  const trace = order.data;
  const steps = trace?.policies.find((p) => p.name === policy)?.steps;

  useEffect(() => {
    if (trace) useReplay.getState().load(trace.order.horizon, policy, !prefersReducedMotion());
  }, [trace, policy]);
  useEffect(() => startReplayLoop(), []);
  const k = useReplay((s) => s.k);

  const path = steps?.regime ?? [];
  const focus = trace?.policies.find((p) => p.name === policy);
  const d = focus ? decisionAt(focus, k) : null;
  const active = activeRule(d?.phi);

  if (meta.error) return <p role="alert" className="flex items-center gap-2 text-sm text-fg-muted"><AlertTriangle aria-hidden className="size-4 text-bad" /> {meta.error.message}</p>;

  return (
    <>
      <PageHeader eyebrow="Brain" title="The fuzzy rules and the regime space">
        <div className="flex flex-wrap items-center gap-2">
          {explainers.length > 1 && (
            <Segmented label="Policy" value={policy ?? ""} onChange={(v) => setPolicy(v)}
              options={explainers.map((p) => ({ value: p.name, label: p.label.replace(" (seed 0)", "") }))} />
          )}
          <Button onClick={() => setSeed(Math.floor(Math.random() * 1e6))} disabled={order.isFetching} title="Read the rules along a different random order">
            <Shuffle aria-hidden className="size-4" /> Another order
          </Button>
        </div>
      </PageHeader>
      {trace && (
        <p className="-mt-5 mb-6 text-sm text-fg-muted">
          Along a {trace.order.side} order in {trace.order.security.replace(" IS Equity", "")} on {trace.order.date}, {trace.order.start}–{trace.order.end}.
          Each rule is an ellipsoid at its centre, ±1σ wide, coloured by its pace (<span style={{ color: paceColor(0.85, theme) }}>slower</span> ·{" "}
          <span style={{ color: paceColor(1.15, theme) }}>faster</span>); the line is the market state the agent saw.
        </p>
      )}

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
        <Panel className="relative h-[560px] overflow-hidden">
          {rules.data && trace ? (
            webgl ? (
              <Suspense fallback={<div className="skeleton m-4 h-[520px]" />}>
                <RegimeSpace rules={rules.data} path={path} k={k} active={active} hover={hover} onHover={setHover} theme={theme} ink={ink} />
              </Suspense>
            ) : (
              <RegimeSpace2D rules={rules.data} path={path} k={k} active={active} hover={hover} onHover={setHover} theme={theme} />
            )
          ) : order.error || rules.error ? (
            <p role="alert" className="grid h-full place-items-center text-sm text-fg-muted">{(order.error ?? rules.error)?.message}</p>
          ) : (
            <div className="skeleton m-4 h-[520px]" aria-label="Loading the regime space" />
          )}
          {webgl && rules.data && trace && (
            <p className="pointer-events-none absolute bottom-3 left-4 text-xs text-fg-subtle">Drag to orbit · scroll to zoom · hover a rule</p>
          )}
        </Panel>

        <Panel className="p-5">
          <Eyebrow>Reasoning now</Eyebrow>
          <div className="mt-3">
            {rules.data && d ? (
              <FuzzyBrain rules={rules.data} phi={d.phi} regime={d.regime} onHover={setHover} />
            ) : (
              <div className="space-y-2"><div className="skeleton h-24" /><div className="skeleton h-32" /><div className="skeleton h-40" /></div>
            )}
          </div>
        </Panel>
      </div>

      {rules.data && (
        <Panel className="mt-5 overflow-x-auto p-5">
          <Eyebrow>The rule base ({rules.data.rules.length} rules)</Eyebrow>
          <table className="mt-3 w-full min-w-[640px] text-sm">
            <caption className="sr-only">Every rule: its conditions, its action and how often it fires on validation orders</caption>
            <thead>
              <tr className="text-left text-xs text-fg-subtle">
                <th scope="col" className="pb-2 font-medium">Rule</th>
                <th scope="col" className="pb-2 font-medium">IF</th>
                <th scope="col" className="pb-2 font-medium">THEN</th>
                <th scope="col" className="pb-2 text-right font-medium" title="Mean firing on validation orders">Share</th>
              </tr>
            </thead>
            <tbody>
              {rules.data.rules.map((r, i) => {
                const s = ruleSentence(r);
                return (
                  <tr key={r.rule} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} data-active={i === active}
                    className="border-t border-hairline transition-colors hover:bg-surface-2 data-[active=true]:bg-surface-2">
                    <th scope="row" className="py-2.5 pr-3 text-left font-mono font-normal">
                      <span className="inline-flex items-center gap-2">
                        <span aria-hidden className="size-2.5 rounded-full" style={{ background: paceColor(r.pace, theme) }} />R{r.rule}
                      </span>
                    </th>
                    <td className="py-2.5 pr-3 text-fg-muted">
                      {s.clauses.map((c, j) => (
                        <span key={c.input}>{j > 0 && " ∧ "}{c.input.toLowerCase()} <span className="text-fg">{c.term}</span></span>
                      ))}
                    </td>
                    <td className="py-2.5 pr-3">{s.then}</td>
                    <td className="tabular py-2.5 text-right font-mono text-fg-muted">{Math.round(r.share * 100)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      )}
      {trace && <TransportDock trace={trace} />}
    </>
  );
}
