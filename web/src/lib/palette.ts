/**
 * Data palette (docs/UI_DESIGN.md §2). Validated with the dataviz skill's validate_palette.js on our chart
 * surfaces, in this order (the order is the colour-blind-safety mechanism; never re-order or cycle):
 *   dark  (#12131a): all checks pass; worst adjacent CVD ΔE 9.4, normal-vision ΔE 19.3, all ≥ 3:1 contrast.
 *   light (#ffffff): separation passes (CVD ΔE 9.2, normal 19.6); slots 2, 5, 6 are below 3:1 contrast, so in
 *   light mode every chart using them must show direct labels or a table view (the relief rule).
 * VWAP is the reference, so it is neutral grey and outside the categorical slots. Text never takes a series
 * colour: values and labels stay in the fg tokens.
 */
export type Theme = "dark" | "light";

const SLOTS: Record<Theme, readonly string[]> = {
  //      violet     aqua       orange     blue       magenta    yellow
  dark: ["#9085e9", "#199e70", "#d95926", "#3987e5", "#d55181", "#c98500"],
  light: ["#6250d6", "#1baf7a", "#eb6834", "#2a78d6", "#e87ba4", "#eda100"],
};

/** Neutral reference series (VWAP). */
export const REFERENCE: Record<Theme, string> = { dark: "#8b8e9b", light: "#6b6e7a" };

/**
 * Fixed series → slot assignment, so a policy keeps its colour on every screen and under every filter
 * (colour follows the entity, never its rank).
 */
const SERIES_SLOT: Record<string, number> = {
  efnn_s0: 0,
  efnn: 0,
  mlp: 1,
  efnn_limit_s0: 2,
  efnn_limit: 2,
  mlp_limit: 3,
  anfis: 4,
  almgren_chriss: 5,
};

/** Baselines other than VWAP that share a slot with a policy are drawn dashed (secondary encoding). */
const BASELINE_SLOT: Record<string, number> = { twap: 3, vwap_passive100: 2 };

export function seriesColor(name: string, theme: Theme): string {
  if (name === "vwap") return REFERENCE[theme];
  const slot = SERIES_SLOT[name] ?? BASELINE_SLOT[name];
  return slot === undefined ? REFERENCE[theme] : (SLOTS[theme][slot] ?? REFERENCE[theme]);
}

export function seriesDashed(name: string): boolean {
  return name in BASELINE_SLOT;
}

/** Diverging pace scale: slow (cool) → on schedule (neutral grey) → fast (warm). */
export const PACE_SCALE: Record<Theme, { slow: string; neutral: string; fast: string }> = {
  dark: { slow: "#3987e5", neutral: "#4a4c57", fast: "#d95926" },
  light: { slow: "#2a78d6", neutral: "#d9dae1", fast: "#eb6834" },
};

export const NEEDS_DIRECT_LABELS_IN_LIGHT = new Set([1, 4, 5]); // slot indices below 3:1 on white
