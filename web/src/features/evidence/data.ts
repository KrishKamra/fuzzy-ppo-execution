/**
 * The Evidence screen's data. Measured numbers come from the API's report tables (runs/reports via
 * configs/demo.yaml); the independent replication's numbers, whose episode rows are not kept, are recorded here
 * from docs/RESULTS.md ("M4 replication and stricter statistics"), the same values as hftf_eval/figures.py.
 */
import type { Results } from "@/api/client";

export type Row = Record<string, string | number | null>;

export function table(results: Results | undefined, id: string): Row[] {
  return (results?.tables.find((t) => t.id === id)?.rows as Row[] | undefined) ?? [];
}

export function row(rows: Row[], policy: string, reference?: string): Row | undefined {
  return rows.find((r) => r.policy === policy && (reference === undefined || r.reference === reference));
}

export const num = (r: Row | undefined, key: string): number | null => {
  const v = r?.[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
};

export const ENSEMBLES = ["mlp", "anfis", "efnn"] as const;
export const LABELS: Record<string, string> = {
  mlp: "PPO-MLP", anfis: "PPO-ANFIS", efnn: "PPO-eFNN", lip_mlp: "MLP + limit orders", lip_efnn: "eFNN + limit orders",
  vwap_passive100: "VWAP, always passive",
}; // fmt: skip

/** Replication (3 seeds, episode seed 42): mean, session CI, date CI, Holm p (docs/RESULTS.md). */
export const REPLICATION_VS_VWAP: Record<string, { mean: number; lo: number; hi: number; dateLo: number; dateHi: number; holm: number }> = {
  mlp: { mean: -0.136, lo: -0.271, hi: -0.007, dateLo: -0.222, dateHi: -0.053, holm: 0.074 },
  anfis: { mean: -0.195, lo: -0.354, hi: -0.049, dateLo: -0.316, dateHi: -0.082, holm: 0.059 },
  efnn: { mean: -0.237, lo: -0.436, hi: -0.067, dateLo: -0.371, dateHi: -0.119, holm: 0.018 },
};
export const REPLICATION_VS_MLP: Record<string, { mean: number; lo: number; hi: number; holm: number }> = {
  anfis: { mean: -0.06, lo: -0.154, hi: 0.024, holm: 1 },
  efnn: { mean: -0.101, lo: -0.212, hi: -0.009, holm: 0.324 },
};

export type Status = "supported" | "partly" | "not";

/** What the demo claims, and what it does not (docs/DEMO_PLAN.md §5). */
export const CLAIMS: { claim: string; status: Status; evidence: string; anchor: string }[] = [
  { claim: "Beats VWAP, TWAP and Almgren-Chriss out of sample", status: "supported",
    evidence: "−0.16 to −0.24 bps on the held-out test split; reproduced by an independent retraining; date-cluster CIs exclude 0", anchor: "vs-vwap" },
  { claim: "Survives replay against the real order book", status: "supported",
    evidence: "TickSim on identical episodes: the edge holds and grows by ≈0.05 bps", anchor: "fills" },
  { claim: "The edge is transient-impact management", status: "supported",
    evidence: "Vanishes without transient impact, grows as it strengthens or persists", anchor: "robustness" },
  { claim: "Learned limit orders beat the naive passive rule", status: "supported",
    evidence: "−0.43 to −0.46 bps vs always-passive, with a lighter tail (CVaR 69 vs 75); fills are counterfactual", anchor: "limit" },
  { claim: "Interpretable rules that follow execution theory", status: "supported",
    evidence: "In 5/5 eFNN seeds calm regimes keep or beat the schedule and stressed regimes slow down", anchor: "brain" },
  { claim: "The fuzzy architecture beats the neural network", status: "not",
    evidence: "No pairwise difference survives Holm in either run (p ≥ 0.32); non-significance is not equivalence", anchor: "architectures" },
]; // fmt: skip
