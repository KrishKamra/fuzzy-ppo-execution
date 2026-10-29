import type { Meta, OrderTrace, PolicyTrace, Rules } from "@/api/client";

export const META: Meta = {
  securities: ["ADE IS Equity", "APHS IS Equity"],
  sessions: [
    { security: "ADE IS Equity", date: "2026-09-29", split: "val", ticks: true },
    { security: "ADE IS Equity", date: "2026-09-08", split: "val", ticks: false },
    { security: "ADE IS Equity", date: "2026-10-05", split: "test", ticks: true },
    { security: "APHS IS Equity", date: "2026-09-29", split: "val", ticks: true },
  ],
  horizons: [30, 60, 120],
  earliest_start: "09:20",
  latest_end: "15:15",
  participation: [0.01, 0.2],
  policies: [
    { name: "efnn_s0", label: "PPO-eFNN (seed 0)", kind: "ppo", description: "", explains: true, limit_orders: false },
    { name: "mlp", label: "PPO-MLP (5-seed ensemble)", kind: "ppo", description: "", explains: false, limit_orders: false },
    { name: "vwap", label: "VWAP", kind: "baseline", description: "", explains: false, limit_orders: false },
    { name: "twap", label: "TWAP", kind: "baseline", description: "", explains: false, limit_orders: false },
  ],
  default_policies: ["efnn_s0", "mlp", "vwap"],
  runs: [],
  regime_inputs: ["imb_bar_proxy_ema5", "log_spread_var_30m", "log_rv_30m"],
};

/** A small, exact trace: 3 context minutes, a 4-minute order, three policies with known costs. */
export function makeTrace(): OrderTrace {
  const context = 3;
  const h = 4;
  const row0 = 100;
  const minutes = Array.from({ length: context + h + 1 }, (_, i) => row0 - context + i);
  const qty = 400;
  const policy = (name: string, label: string, cost: number[], q: number[], passive = 0): PolicyTrace => {
    let left = qty;
    const remaining = q.map((x) => (left -= x));
    return {
      name, label,
      summary: { cost_bps: cost[cost.length - 1] ?? 0, is_bps: 0, fees_bps: 0, risk_bps: 0, vwap_slip_bps: 0, n_children: q.filter((x) => x > 0).length, executed: qty, passive_frac: 0 },
      steps: {
        minute: Array.from({ length: h }, (_, i) => row0 + 1 + i),
        pace: q.map((x) => x / 100), passive: q.map(() => passive), q,
        posted: q.map((x) => x * passive), passive_filled: q.map((x) => x * passive),
        price: q.map((_, i) => 1000 + i), remaining,
        cum_is_bps: cost, cum_fees_bps: cost.map(() => 0), cum_risk_bps: cost.map(() => 0), cum_cost_bps: cost,
        regime: q.map(() => [0.5, -1, 2]),
        phi: name === "efnn_s0" ? q.map(() => [0.7, 0.2, 0.1]) : null,
      },
    };
  }; // fmt: skip
  return {
    order: { security: "ADE IS Equity", date: "2026-09-29", split: "val", start: "11:00", end: "11:04", horizon: h, side: "buy", qty, arrival_mid: 1000, sim: "bar" },
    market: {
      minute: minutes,
      time: minutes.map((m) => { const t = 9 * 60 + 15 + m; return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`; }),
      mid: minutes.map((_, i) => 1000 + i), bid: minutes.map((_, i) => 999.9 + i), ask: minutes.map((_, i) => 1000.1 + i),
      volume: minutes.map(() => 500), expected_volume: minutes.map(() => 500),
    },
    regime_inputs: META.regime_inputs,
    policies: [
      policy("efnn_s0", "PPO-eFNN (seed 0)", [1, 2, 3, 4], [100, 100, 100, 100]),
      policy("mlp", "PPO-MLP (5-seed ensemble)", [2, 3, 3.5, 6], [150, 50, 0, 200]),
      policy("vwap", "VWAP", [1.5, 3, 4, 5], [100, 100, 100, 100]),
    ],
  }; // fmt: skip
}

/** Three rules matching the fixture trace's three-way phi (R0 fires 70%). */
export const RULES: Rules = {
  policy: "efnn_s0",
  inputs: ["imbalance", "spread_var", "rv"],
  input_labels: ["Order-book imbalance", "Spread variance", "Realised vol"],
  terciles: [[-0.4, 0.4], [-0.5, 0.5], [-0.45, 0.45]],
  rules: [
    { rule: 0, centres: [0, -1, 0.2], widths: [1, 0.8, 0.9], terms: ["NEUTRAL", "LOW", "MED"], pace: 0.97, passive: null, share: 0.8 },
    { rule: 1, centres: [0.1, -1.2, -1.1], widths: [1, 0.7, 0.6], terms: ["NEUTRAL", "LOW", "LOW"], pace: 1.06, passive: null, share: 0.12 },
    { rule: 2, centres: [0, 1.4, 1.5], widths: [1.1, 0.9, 0.9], terms: ["NEUTRAL", "HIGH", "HIGH"], pace: 0.88, passive: null, share: 0.08 },
  ],
};
