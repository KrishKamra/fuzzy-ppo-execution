/**
 * The Execute screen's single replay clock. `pos` is fractional (smooth timing); `k = floor(pos)` is the number
 * of decisions revealed, 0 … horizon, and is what every panel renders from, so they can never disagree.
 */
import { create } from "zustand";

export const STEPS_PER_SECOND = 2; // at 1×: a 60-minute order replays in 30 s
export const SPEEDS = [1, 4, 16] as const;
export type Speed = (typeof SPEEDS)[number];

interface ReplayState {
  horizon: number;
  pos: number;
  k: number;
  playing: boolean;
  speed: Speed;
  focus: string | null; // the policy whose fills and decisions are shown in detail
  load: (horizon: number, focus: string | null, autoplay: boolean) => void;
  advance: (seconds: number) => void;
  seek: (k: number) => void;
  step: (delta: number) => void;
  toggle: () => void;
  setSpeed: (speed: Speed) => void;
  setFocus: (name: string) => void;
}

const clamp = (x: number, hi: number) => Math.max(0, Math.min(hi, x));

export const useReplay = create<ReplayState>((set, get) => ({
  horizon: 0,
  pos: 0,
  k: 0,
  playing: false,
  speed: 1,
  focus: null,
  load: (horizon, focus, autoplay) =>
    set({ horizon, focus, pos: autoplay ? 0 : horizon, k: autoplay ? 0 : horizon, playing: autoplay }),
  advance: (seconds) => {
    const { pos, horizon, speed, playing } = get();
    if (!playing) return;
    const next = clamp(pos + seconds * STEPS_PER_SECOND * speed, horizon);
    set({ pos: next, k: Math.floor(next), playing: next < horizon });
  },
  seek: (k) => {
    const v = clamp(Math.round(k), get().horizon);
    set({ pos: v, k: v, playing: false });
  },
  step: (delta) => get().seek(get().k + delta),
  toggle: () => {
    const { playing, k, horizon } = get();
    // Pressing play at the end restarts the replay.
    if (!playing && k >= horizon) set({ pos: 0, k: 0, playing: true });
    else set({ playing: !playing });
  },
  setSpeed: (speed) => set({ speed }),
  setFocus: (name) => set({ focus: name }),
}));

/** Drive the clock with requestAnimationFrame while playing. */
export function startReplayLoop(): () => void {
  let last = performance.now();
  let frame = 0;
  const tick = (now: number) => {
    useReplay.getState().advance(Math.min((now - last) / 1000, 0.1)); // cap a backgrounded tab's jump
    last = now;
    frame = requestAnimationFrame(tick);
  };
  frame = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(frame);
}
