import { Command } from "cmdk";
import { Moon, Presentation, Shuffle } from "lucide-react";
import { useNavigate } from "react-router";

import { NAV } from "@/app/nav";
import { Kbd } from "@/components/ui/primitives";
import { useUi } from "@/store/ui";

const ITEM =
  "flex cursor-pointer items-center gap-3 rounded-[var(--radius-control)] px-3 py-2.5 text-sm text-fg-muted " +
  "data-[selected=true]:bg-surface-3 data-[selected=true]:text-fg";

/** Ctrl/⌘ K: jump anywhere, flip modes, start a random order (docs/UI_DESIGN.md §3, global). */
export function CommandPalette() {
  const { paletteOpen, setPaletteOpen, toggleTheme, togglePresentation } = useUi();
  const navigate = useNavigate();
  const run = (fn: () => void) => () => {
    setPaletteOpen(false);
    fn();
  };
  return (
    <Command.Dialog
      open={paletteOpen}
      onOpenChange={setPaletteOpen}
      label="Command palette"
      overlayClassName="fixed inset-0 z-50 bg-black/40"
      contentClassName="glass fixed top-[18vh] left-1/2 z-50 w-[min(560px,calc(100vw-32px))] -translate-x-1/2 overflow-hidden rounded-xl border border-hairline-strong shadow-2xl"
    >
      <Command.Input
        placeholder="Go to a screen or run a command…"
        className="w-full border-b border-hairline bg-transparent px-4 py-3.5 text-[0.95rem] text-fg outline-none placeholder:text-fg-subtle"
      />
      <Command.List className="max-h-[50vh] overflow-y-auto p-2">
        <Command.Empty className="px-3 py-6 text-center text-sm text-fg-subtle">No matches</Command.Empty>
        <Command.Group heading="Screens" className="text-xs text-fg-subtle [&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-2">
          {NAV.map((item, i) => (
            <Command.Item key={item.path} value={`${item.label} ${item.hint}`} onSelect={run(() => navigate(item.path))} className={ITEM}>
              <item.icon aria-hidden className="size-4" />
              <span className="flex-1">
                <span className="text-fg">{item.label}</span>
                <span className="ml-2 text-fg-subtle">{item.hint}</span>
              </span>
              <Kbd>{i + 1}</Kbd>
            </Command.Item>
          ))}
        </Command.Group>
        <Command.Group heading="Actions" className="text-xs text-fg-subtle [&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-2">
          <Command.Item value="random order surprise me" onSelect={run(() => navigate("/execute?random=1"))} className={ITEM}>
            <Shuffle aria-hidden className="size-4" />
            <span className="flex-1 text-fg">Run a random order</span>
          </Command.Item>
          <Command.Item value="presentation mode projector" onSelect={run(togglePresentation)} className={ITEM}>
            <Presentation aria-hidden className="size-4" />
            <span className="flex-1 text-fg">Toggle presentation mode</span>
            <Kbd>P</Kbd>
          </Command.Item>
          <Command.Item value="theme dark light" onSelect={run(toggleTheme)} className={ITEM}>
            <Moon aria-hidden className="size-4" />
            <span className="flex-1 text-fg">Toggle theme</span>
            <Kbd>T</Kbd>
          </Command.Item>
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}
