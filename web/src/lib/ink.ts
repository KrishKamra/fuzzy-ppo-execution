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
    theme: "dark", fg: "rgba(255,255,255,0.92)", muted: "rgba(255,255,255,0.64)", subtle: "rgba(255,255,255,0.5)",
    hairline: "rgba(255,255,255,0.08)", surface: "#12131a", surface2: "#181a23", accent: "#8b5cf6",
  },
  light: {
    theme: "light", fg: "rgba(12,14,24,0.92)", muted: "rgba(12,14,24,0.7)", subtle: "rgba(12,14,24,0.6)",
    hairline: "rgba(15,18,30,0.09)", surface: "#ffffff", surface2: "#f1f2f7", accent: "#8b5cf6",
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
