/**
 * Pure views of an order trace at replay position k (decisions revealed). Everything the Execute screen shows is
 * derived here, so the panels agree by construction and the logic is testable without a canvas.
 */
import type { OrderTrace, PolicyTrace } from "@/api/client";

export const REFERENCE = "vwap";

/** "PPO-MLP (5-seed ensemble)" → ["PPO-MLP", "5-seed ensemble"]: the name fits a narrow column, the qualifier
 *  goes on the line below. Labels without a trailing parenthesis come back whole. */
export function splitLabel(label: string): [string, string | null] {
  const m = /^(.*?)\s*\(([^()]*)\)$/.exec(label);
  return m ? [m[1] ?? label, m[2] ?? null] : [label, null];
}

/** Index of a minute on the shared x axis (the trace's market minutes). */
export function axisIndex(trace: OrderTrace, minute: number): number {
  return minute - (trace.market.minute[0] ?? 0);
}

/** The x index where the order starts (the decision before the first fill bar). */
export function startIndex(trace: OrderTrace): number {
  const first = trace.policies[0]?.steps.minute[0];
  return first === undefined ? 0 : axisIndex(trace, first) - 1;
}

/** The x index of the replay cursor after k decisions. */
export function cursorIndex(trace: OrderTrace, k: number): number {
  return startIndex(trace) + k;
}

export function reference(trace: OrderTrace): PolicyTrace | undefined {
  return trace.policies.find((p) => p.name === REFERENCE);
}

/** A per-minute series for one policy over the shared axis: null before the order and after the cursor. */
function overAxis(trace: OrderTrace, k: number, values: (i: number) => number, atStart: number): (number | null)[] {
  const out: (number | null)[] = Array(trace.market.minute.length).fill(null);
  const s = startIndex(trace);
  out[s] = atStart;
  for (let i = 0; i < k; i++) out[s + 1 + i] = values(i);
  return out;
}

/** % of the order still to execute, revealed up to k. */
export function inventorySeries(trace: OrderTrace, p: PolicyTrace, k: number): (number | null)[] {
  const qty = trace.order.qty;
  return overAxis(trace, k, (i) => (100 * (p.steps.remaining[i] ?? 0)) / qty, 100);
}

/** Running cost minus the reference's running cost (bps; below zero = cheaper than VWAP), revealed up to k. */
export function costRaceSeries(trace: OrderTrace, p: PolicyTrace, k: number): (number | null)[] {
  const ref = reference(trace);
  return overAxis(trace, k, (i) => (p.steps.cum_cost_bps[i] ?? 0) - (ref?.steps.cum_cost_bps[i] ?? 0), 0);
}

/** Market prices revealed up to the cursor (the pre-order context is always shown). */
export function marketSeries(trace: OrderTrace, k: number, key: "mid" | "bid" | "ask"): (number | null)[] {
  const end = cursorIndex(trace, k);
  return trace.market[key].map((v, i) => (i <= end ? v : null));
}

export interface Fill {
  index: number;
  price: number;
  q: number;
  passive: number;
  time: string;
}

/** The fills of one policy up to k: one point per non-empty child, at its fill bar. */
export function fills(trace: OrderTrace, p: PolicyTrace, k: number): Fill[] {
  const out: Fill[] = [];
  for (let i = 0; i < k; i++) {
    const q = p.steps.q[i] ?? 0;
    if (q <= 0) continue;
    const minute = p.steps.minute[i] ?? 0;
    const index = axisIndex(trace, minute);
    out.push({ index, price: p.steps.price[i] ?? 0, q, passive: p.steps.passive_filled[i] ?? 0, time: trace.market.time[index] ?? "" });
  }
  return out;
}

export interface Standing {
  name: string;
  label: string;
  costSoFar: number; // bps of arrival notional
  vsReference: number | null;
  done: number; // share of the order executed
  children: number;
  passiveShare: number;
  finished: boolean;
  finalCost: number | null;
}

/** The leaderboard after k decisions, best (most negative vs VWAP) first; VWAP itself is listed last. */
export function standings(trace: OrderTrace, k: number): Standing[] {
  const ref = reference(trace);
  const qty = trace.order.qty;
  const at = (arr: number[], i: number) => (i >= 0 ? (arr[i] ?? 0) : 0);
  const rows = trace.policies.map((p) => {
    const s = p.steps;
    const n = s.q.length;
    const last = Math.min(k, n) - 1;
    const costSoFar = at(s.cum_cost_bps, last);
    let children = 0;
    let passive = 0;
    for (let i = 0; i <= last; i++) {
      if ((s.q[i] ?? 0) > 0) children++;
      passive += s.passive_filled[i] ?? 0;
    }
    const finished = k >= n;
    return {
      name: p.name,
      label: p.label,
      costSoFar,
      vsReference: ref && p.name !== REFERENCE ? costSoFar - at(ref.steps.cum_cost_bps, last) : null,
      done: last < 0 ? 0 : 1 - at(s.remaining, last) / qty,
      children,
      passiveShare: passive / qty,
      finished,
      finalCost: finished ? (p.summary.cost_bps ?? null) : null,
    };
  });
  const key = (r: Standing) => (r.name === REFERENCE ? Infinity : (r.vsReference ?? r.costSoFar));
  return rows.sort((a, b) => key(a) - key(b));
}

/** The focus policy's decision at the cursor: the one just executed (k ≥ 1), else the first. */
export function decisionAt(p: PolicyTrace, k: number) {
  const i = Math.max(0, Math.min(k, p.steps.q.length) - 1);
  return {
    index: i,
    minute: p.steps.minute[i] ?? 0,
    pace: p.steps.pace[i] ?? 1,
    passive: p.steps.passive[i] ?? 0,
    q: p.steps.q[i] ?? 0,
    regime: p.steps.regime[i] ?? [],
    phi: p.steps.phi?.[i] ?? null,
  };
}
