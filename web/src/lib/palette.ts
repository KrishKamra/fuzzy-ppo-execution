/**
 * Data palette (docs/UI_DESIGN.md §2, theme "Desk"). Slot 0 is amber, the agent's colour (eFNN). Validated with the
 * dataviz skill's validate_palette.js on our chart surfaces, in this order (the order is the colour-blind-safety
 * mechanism; never re-order or cycle):
 *   dark  (#171a1f): all checks pass; worst adjacent CVD ΔE 9.4, normal-vision ΔE 19.7, all ≥ 3:1 contrast.
 *   light (#ffffff): separation passes (CVD ΔE 9.2, normal 27.6); slots 1, 3, 6 are below 3:1 contrast, so in
 *   light mode every chart using them must show direct labels or a table view (the relief rule).
 * VWAP is the reference, so it is neutral grey and outside the categorical slots. Text never takes a series
 * colour: values and labels stay in the fg tokens.
 */
export type Theme = "dark" | "light";

const SLOTS: Record<Theme, readonly string[]> = {
  //      amber      blue       aqua       orange     violet     magenta
  dark: ["#c98500", "#3987e5", "#199e70", "#d95926", "#9085e9", "#d55181"],
  light: ["#eda100", "#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#e87ba4"],
};

/** Neutral reference series (VWAP). */
export const REFERENCE: Record<Theme, string> = { dark: "#8b919b", light: "#6b717b" };

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
  dark: { slow: "#3987e5", neutral: "#4a4f58", fast: "#d95926" },
  light: { slow: "#2a78d6", neutral: "#d8dbe0", fast: "#eb6834" },
};

export const NEEDS_DIRECT_LABELS_IN_LIGHT = new Set([0, 2, 5]); // slot indices below 3:1 on white
