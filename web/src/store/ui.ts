/**
 * App-wide UI state: theme, presentation mode (projector), command palette. Theme and presentation are
 * per-viewer conveniences kept in localStorage; storage can be unavailable (private windows, blocked site
 * data), so every access is guarded and the app works without it.
 */
import { create } from "zustand";

import type { Theme } from "@/lib/palette";

const KEY = "hftf.ui";

interface Persisted {
  theme: Theme;
  presentation: boolean;
}

function load(): Persisted {
  try {
    const raw = window.localStorage.getItem(KEY);
    const v = raw ? (JSON.parse(raw) as Partial<Persisted>) : {};
    return { theme: v.theme === "light" ? "light" : "dark", presentation: v.presentation === true };
  } catch {
    return { theme: "dark", presentation: false };
  }
}

function save(v: Persisted): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(v));
  } catch {
    // storage blocked: the setting simply does not persist
  }
}

/** Reflect the state on <html>, where the CSS tokens read it. Presentation mode forces the light theme. */
export function applyToDocument(v: Persisted): void {
  const root = document.documentElement;
  const theme = v.presentation ? "light" : v.theme;
  root.dataset.theme = theme;
  if (v.presentation) root.dataset.presentation = "";
  else delete root.dataset.presentation;
}

interface UiState extends Persisted {
  paletteOpen: boolean;
  effectiveTheme: () => Theme;
  toggleTheme: () => void;
  togglePresentation: () => void;
  setPaletteOpen: (open: boolean) => void;
}

export const useUi = create<UiState>((set, get) => {
  const initial = load();
  const commit = (next: Persisted) => {
    save(next);
    applyToDocument(next);
    set(next);
  };
  return {
    ...initial,
    paletteOpen: false,
    effectiveTheme: () => (get().presentation ? "light" : get().theme),
    toggleTheme: () => commit({ theme: get().theme === "dark" ? "light" : "dark", presentation: get().presentation }),
    togglePresentation: () => commit({ theme: get().theme, presentation: !get().presentation }),
    setPaletteOpen: (open) => set({ paletteOpen: open }),
  };
});
