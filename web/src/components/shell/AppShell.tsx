import { useEffect } from "react";
import { Outlet, useNavigate } from "react-router";

import { NAV } from "@/app/nav";
import { isTyping } from "@/lib/dom";
import { applyToDocument, useUi } from "@/store/ui";

import { CommandPalette } from "./CommandPalette";
import { TopBar } from "./TopBar";

/** Global keys: Ctrl/⌘ K palette, 1-5 screens, P presentation, T theme. Ignored while typing in a field. */
function useGlobalHotkeys() {
  const navigate = useNavigate();
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const ui = useUi.getState();
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        ui.setPaletteOpen(!ui.paletteOpen);
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey || ui.paletteOpen || isTyping(e.target)) return;
      const index = Number.parseInt(e.key, 10) - 1;
      const item = NAV[index];
      if (item) navigate(item.path);
      else if (e.key.toLowerCase() === "p") ui.togglePresentation();
      else if (e.key.toLowerCase() === "t" && !ui.presentation) ui.toggleTheme();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [navigate]);
}

/** Two soft aurora glows behind everything, at a few percent opacity: ambience, never under-chart contrast. */
function AmbientBackground() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div
        className="absolute -top-[30vh] left-[10vw] size-[70vh] rounded-full blur-[120px]"
        style={{ background: "var(--accent-2)", opacity: "var(--ambient-opacity)" }}
      />
      <div
        className="absolute top-[20vh] -right-[15vw] size-[60vh] rounded-full blur-[120px]"
        style={{ background: "var(--accent-3)", opacity: "var(--ambient-opacity)" }}
      />
    </div>
  );
}

export function AppShell() {
  const { theme, presentation } = useUi();
  useEffect(() => applyToDocument({ theme, presentation }), [theme, presentation]);
  useGlobalHotkeys();
  return (
    <div className="relative isolate min-h-dvh">
      <AmbientBackground />
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:bg-surface-2 focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <TopBar />
      <main id="main" className="mx-auto max-w-[1440px] px-4 pt-10 pb-24 sm:px-6">
        <Outlet />
      </main>
      <CommandPalette />
    </div>
  );
}
