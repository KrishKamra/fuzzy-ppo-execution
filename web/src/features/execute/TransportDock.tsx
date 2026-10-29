import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useEffect } from "react";

import type { OrderTrace } from "@/api/client";
import { Button, Kbd, Segmented } from "@/components/ui/primitives";
import { isTyping } from "@/lib/dom";
import { SPEEDS, type Speed, useReplay } from "@/store/replay";

import { cursorIndex } from "./model";

/** Space plays/pauses, ← → step one decision (Shift: ten). Only while the Execute screen is mounted. */
function useTransportKeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || isTyping(e.target)) return;
      const r = useReplay.getState();
      if (e.key === " ") {
        // A focused button already acts on Space; handling it here too would toggle twice.
        if ((e.target as HTMLElement | null)?.closest?.("button, [role='radio'], a")) return;
        e.preventDefault();
        r.toggle();
      } else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        r.step((e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 10 : 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

/** Floating glass dock: the replay's transport, scrubber and clock. */
export function TransportDock({ trace }: { trace: OrderTrace }) {
  useTransportKeys();
  const { k, horizon, playing, speed, toggle, step, seek, setSpeed } = useReplay();
  const time = trace.market.time[cursorIndex(trace, k)] ?? trace.order.start;
  return (
    <div className="glass fixed inset-x-0 bottom-4 z-30 mx-auto flex w-[min(880px,calc(100vw-32px))] items-center gap-3 rounded-2xl border border-hairline-strong px-3 py-2 shadow-2xl">
      <Button variant="ghost" className="px-2" onClick={() => step(-1)} aria-label="Back one decision (←)" disabled={k === 0}>
        <SkipBack className="size-4" />
      </Button>
      <Button variant="primary" className="size-9 justify-center px-0" onClick={toggle} aria-label={playing ? "Pause (Space)" : "Play (Space)"}>
        {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
      </Button>
      <Button variant="ghost" className="px-2" onClick={() => step(1)} aria-label="Forward one decision (→)" disabled={k >= horizon}>
        <SkipForward className="size-4" />
      </Button>
      <div className="tabular w-[4.5rem] shrink-0 text-center font-mono text-sm" aria-live="off">
        {time}
      </div>
      <input
        type="range"
        aria-label="Replay position (decisions)"
        aria-valuetext={`${time}, decision ${k} of ${horizon}`}
        min={0}
        max={horizon}
        step={1}
        value={k}
        onChange={(e) => seek(Number(e.target.value))}
        className="min-w-0 flex-1 accent-[var(--accent-2)]"
      />
      <span className="tabular hidden w-14 shrink-0 text-right font-mono text-xs text-fg-subtle sm:inline">
        {k}/{horizon}
      </span>
      <Segmented<Speed> label="Replay speed" value={speed} onChange={setSpeed} options={SPEEDS.map((s) => ({ value: s, label: `${s}×` }))} />
      <span className="hidden items-center gap-1 text-xs text-fg-subtle lg:inline-flex">
        <Kbd>Space</Kbd>
        <Kbd>← →</Kbd>
      </span>
    </div>
  );
}
