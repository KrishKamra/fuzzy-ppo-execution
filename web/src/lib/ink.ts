/**
 * The design tokens charts need, resolved from CSS (ECharts draws on a canvas, so it cannot read CSS
 * variables itself). Re-read whenever the theme changes.
 */
import { useMemo } from "react";

import type { Theme } from "./palette";

export interface Ink {
  theme: Theme;
  fg: string;
  muted: string;
  subtle: string;
  hairline: string;
  surface: string;
  surface2: string;
  accent: string;
}

const FALLBACK: Record<Theme, Ink> = {
  dark: {
    theme: "dark", fg: "rgba(236,240,245,0.94)", muted: "rgba(236,240,245,0.66)", subtle: "rgba(236,240,245,0.52)",
    hairline: "rgba(220,228,240,0.08)", surface: "#171a1f", surface2: "#1d2127", accent: "#f2a93b",
  },
  light: {
    theme: "light", fg: "rgba(21,24,29,0.94)", muted: "rgba(21,24,29,0.7)", subtle: "rgba(21,24,29,0.6)",
    hairline: "rgba(21,24,29,0.1)", surface: "#ffffff", surface2: "#f7f8f9", accent: "#9a5b00",
  },
}; // fmt: skip

export function readInk(theme: Theme): Ink {
  if (typeof window === "undefined") return FALLBACK[theme];
  const css = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  const f = FALLBACK[theme];
  return {
    theme,
    fg: v("--fg", f.fg),
    muted: v("--fg-muted", f.muted),
    subtle: v("--fg-subtle", f.subtle),
    hairline: v("--hairline", f.hairline),
    surface: v("--surface-1", f.surface),
    surface2: v("--surface-2", f.surface2),
    accent: v("--accent-2", f.accent),
  };
}

export function useInk(theme: Theme): Ink {
  return useMemo(() => readInk(theme), [theme]);
}
