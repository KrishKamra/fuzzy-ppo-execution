/**
 * Pure helpers for showing the eFNN's reasoning (ARCHITECTURE §5): Gaussian membership, the rule that fires
 * most, and a rule as a sentence. Inputs are train-standardised; the imbalance input is signed by side, so
 * HIGH means "with us" (the book leans the way we trade).
 */
import type { Rules } from "@/api/client";
import { PACE_SCALE, type Theme } from "@/lib/palette";

export type Rule = Rules["rules"][number];

export const INPUT_NAMES = ["Imbalance", "Spread variance", "Volatility"] as const;

const TERM_WORDS: Record<string, string> = {
  AGAINST: "against us", NEUTRAL: "neutral", WITH: "with us", LOW: "low", MED: "medium", HIGH: "high",
}; // fmt: skip

/** Membership of x in a Gaussian fuzzy set with centre c and width w (σ). */
export function gaussian(x: number, c: number, w: number): number {
  const z = (x - c) / Math.max(w, 1e-6);
  return Math.exp(-0.5 * z * z);
}

/** The rule with the largest normalised firing φ (the "active" regime). */
export function activeRule(phi: readonly number[] | null | undefined): number | null {
  if (!phi || phi.length === 0) return null;
  let best = 0;
  for (let r = 1; r < phi.length; r++) if ((phi[r] ?? 0) > (phi[best] ?? 0)) best = r;
  return best;
}

export function paceVerdict(pace: number): "slow down" | "keep to schedule" | "speed up" {
  return pace < 0.95 ? "slow down" : pace > 1.05 ? "speed up" : "keep to schedule";
}

/** "IF imbalance neutral ∧ spread variance low ∧ volatility medium THEN pace 0.97 (keep to schedule)". */
export function ruleSentence(rule: Rule): { clauses: { input: string; term: string }[]; then: string } {
  const clauses = rule.terms.map((t, j) => ({ input: INPUT_NAMES[j] ?? `x${j}`, term: TERM_WORDS[t] ?? t.toLowerCase() }));
  const passive = rule.passive != null ? `, post ${Math.round(rule.passive * 100)}% passively` : "";
  return { clauses, then: `pace ${rule.pace.toFixed(2)}× (${paceVerdict(rule.pace)})${passive}` };
}

function hexToRgb(hex: string): [number, number, number] {
  const n = Number.parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** A rule's colour from its pace on the diverging scale: slow (cool) → schedule (neutral) → fast (warm). */
export function paceColor(pace: number, theme: Theme): string {
  const s = PACE_SCALE[theme];
  const t = Math.max(-1, Math.min(1, (pace - 1) / 0.1)); // ±10% spans the scale (learned paces lie in 0.85-1.1)
  const [a, b] = t < 0 ? [s.neutral, s.slow] : [s.neutral, s.fast];
  const [r1, g1, b1] = hexToRgb(a);
  const [r2, g2, b2] = hexToRgb(b);
  const u = Math.abs(t);
  const mix = (x: number, y: number) => Math.round(x + (y - x) * u);
  return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`;
}
