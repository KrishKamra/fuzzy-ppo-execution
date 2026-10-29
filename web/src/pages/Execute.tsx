import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Loader2, MousePointerClick } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router";

import { api, type OrderTrace } from "@/api/client";
import { useMeta } from "@/api/queries";
import { Eyebrow, PageHeader, Panel } from "@/components/ui/primitives";
import { DecisionCard } from "@/features/execute/DecisionCard";
import { ExecuteCharts } from "@/features/execute/ExecuteCharts";
import { Leaderboard } from "@/features/execute/Leaderboard";
import { REFERENCE } from "@/features/execute/model";
import { OrderTicket } from "@/features/execute/OrderTicket";
import { defaultTicket, fromSearch, type Ticket, toRequest, toSearch, validate } from "@/features/execute/ticket";
import { TransportDock } from "@/features/execute/TransportDock";
import { startReplayLoop, useReplay } from "@/store/replay";

const prefersReducedMotion = () => typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** The policy shown in detail: the first one that explains itself (a single fuzzy checkpoint), else the first PPO. */
function defaultFocus(trace: OrderTrace): string | null {
  const ppo = trace.policies.filter((p) => p.name !== REFERENCE);
  return (ppo.find((p) => p.steps.phi) ?? ppo[0] ?? trace.policies[0])?.name ?? null;
}

function useOrderTrace(ticket: Ticket | null, valid: boolean) {
  const key = ticket ? toSearch(ticket).toString() : "";
  return useQuery({
    queryKey: ["order", key],
    queryFn: () => {
      const t = ticket as Ticket;
      // VWAP is always in the request: it is the reference every cost is compared with.
      const policies = t.policies.includes(REFERENCE) ? t.policies : [...t.policies, REFERENCE];
      return api.order(toRequest({ ...t, policies }));
    },
    enabled: !!ticket && valid,
    staleTime: Infinity,
    retry: false,
  });
}

function title(trace: OrderTrace | undefined): string {
  if (!trace) return "Replay a real order";
  const o = trace.order;
  return `${o.side === "buy" ? "Buy" : "Sell"} ${Math.round(o.qty).toLocaleString()} ${o.security.replace(" IS Equity", "")} · ${o.start}–${o.end}`;
}

export function Execute() {
  const meta = useMeta();
  const [params, setParams] = useSearchParams();
  const submitted = useMemo(() => fromSearch(params), [params]);
  const [draft, setDraft] = useState<Ticket | null>(null);
  const [showTest, setShowTest] = useState(false);
  const [randomError, setRandomError] = useState<string | null>(null);
  const [randomBusy, setRandomBusy] = useState(false);

  // The form starts from the URL's ticket, else the defaults.
  useEffect(() => {
    if (!meta.data || draft) return;
    const start = submitted ?? defaultTicket(meta.data);
    setDraft(start);
    if (meta.data.sessions.find((s) => s.security === start.security && s.date === start.date)?.split === "test") setShowTest(true);
  }, [meta.data, submitted, draft]);

  const valid = !!meta.data && !!submitted && Object.keys(validate(submitted, meta.data)).length === 0;
  const order = useOrderTrace(submitted, valid);

  const run = useCallback((t: Ticket) => setParams(toSearch(t)), [setParams]);
  const randomize = useCallback(async () => {
    if (!draft) return;
    setRandomBusy(true);
    setRandomError(null);
    try {
      const r = await api.randomTicket(showTest ? (Math.random() < 0.5 ? "val" : "test") : "val", draft.sim, Math.floor(Math.random() * 1e9));
      const t: Ticket = { ...draft, ...r, sim: draft.sim, policies: draft.policies };
      setDraft(t);
      setParams(toSearch(t), { replace: params.has("random") });
    } catch (e) {
      setRandomError((e as Error).message);
    } finally {
      setRandomBusy(false);
    }
  }, [draft, showTest, setParams, params]);

  // ?random=1 (the Overview's "Run a live order"): draw a ticket once the form is ready.
  const randomRequested = params.get("random") === "1";
  const randomFired = useRef(false);
  useEffect(() => {
    if (randomRequested && draft && !randomFired.current) {
      randomFired.current = true;
      void randomize();
    }
  }, [randomRequested, draft, randomize]);

  // A new trace resets the replay (autoplay unless the viewer prefers reduced motion).
  useEffect(() => {
    if (!order.data) return;
    const h = order.data.order.horizon;
    useReplay.getState().load(h, defaultFocus(order.data), !prefersReducedMotion());
  }, [order.data]);
  useEffect(() => startReplayLoop(), []);

  if (meta.isPending) return <p className="flex items-center gap-2 text-sm text-fg-muted"><Loader2 aria-hidden className="size-4 animate-spin" /> Loading sessions…</p>;
  if (meta.error) return <p role="alert" className="flex items-center gap-2 text-sm text-fg-muted"><AlertTriangle aria-hidden className="size-4 text-bad" /> {meta.error.message}</p>;

  const trace = order.data;
  return (
    <>
      <PageHeader eyebrow={trace ? `Execute · ${trace.order.date} · ${trace.order.sim === "tick" ? "TickSim" : "BarSim"}${trace.order.split === "test" ? " · test (out of sample)" : ""}` : "Execute"} title={title(trace)} />
      <div className="grid gap-5 lg:grid-cols-[300px_minmax(0,1fr)] xl:grid-cols-[300px_minmax(0,1fr)_320px]">
        <Panel className="h-fit p-5">
          {draft && (
            <OrderTicket meta={meta.data} ticket={draft} onChange={setDraft} onRun={() => run(draft)} onRandom={() => void randomize()}
              busy={order.isFetching || randomBusy} showTest={showTest} onShowTest={setShowTest} />
          )}
          {randomError && <p role="alert" className="mt-3 text-xs text-bad">{randomError}</p>}
        </Panel>

        <Panel className="min-h-[640px] self-start p-4">
          {order.isFetching && !trace ? (
            <div className="space-y-3" aria-busy="true">
              <p className="flex items-center gap-2 px-2 text-sm text-fg-muted"><Loader2 aria-hidden className="size-4 animate-spin" /> Running every policy on the same order…</p>
              <div className="skeleton h-[250px]" />
              <div className="skeleton h-[140px]" />
              <div className="skeleton h-[180px]" />
            </div>
          ) : order.error ? (
            <div role="alert" className="grid h-full min-h-[600px] place-items-center text-center text-sm text-fg-muted">
              <span><AlertTriangle aria-hidden className="mx-auto mb-2 size-5 text-bad" />{order.error.message}</span>
            </div>
          ) : trace ? (
            <ExecuteCharts trace={trace} />
          ) : (
            <div className="grid h-full min-h-[600px] place-items-center text-center">
              <div className="max-w-sm">
                <MousePointerClick aria-hidden className="mx-auto size-6 text-accent-2" />
                <p className="mt-3 font-medium">Pick an order and press Run</p>
                <p className="mt-1 text-sm text-fg-muted">Every selected policy executes the same order on the same market, minute by minute, against VWAP.</p>
              </div>
            </div>
          )}
        </Panel>

        {trace && (
          <div className="space-y-5 lg:col-span-2 xl:col-span-1">
            <Panel className="p-5">
              <Eyebrow>Standings</Eyebrow>
              <div className="mt-3"><Leaderboard trace={trace} /></div>
            </Panel>
            <Panel className="p-5">
              <Eyebrow>Decision now</Eyebrow>
              <div className="mt-3"><DecisionCard trace={trace} /></div>
            </Panel>
          </div>
        )}
      </div>
      {trace && <TransportDock trace={trace} />}
    </>
  );
}
