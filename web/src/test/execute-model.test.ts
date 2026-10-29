import { costRaceSeries, cursorIndex, decisionAt, fills, inventorySeries, marketSeries, standings, startIndex } from "@/features/execute/model";
import { defaultTicket, fromSearch, latestStart, toSearch, validate } from "@/features/execute/ticket";
import { useReplay } from "@/store/replay";

import { makeTrace, META } from "./fixtures";

test("the shared axis puts the order start after the context and the cursor k decisions later", () => {
  const t = makeTrace();
  expect(startIndex(t)).toBe(3); // three context minutes, then the arrival row
  expect(cursorIndex(t, 0)).toBe(3);
  expect(cursorIndex(t, 4)).toBe(7);
  expect(marketSeries(t, 1, "mid")).toEqual([1000, 1001, 1002, 1003, 1004, null, null, null]);
});

test("series reveal progressively and are relative to VWAP", () => {
  const t = makeTrace();
  const [efnn, mlp] = t.policies;
  expect(inventorySeries(t, efnn!, 2)).toEqual([null, null, null, 100, 75, 50, null, null]);
  expect(costRaceSeries(t, efnn!, 4).slice(3)).toEqual([0, -0.5, -1, -1, -1]);
  expect(fills(t, mlp!, 4).map((f) => f.q)).toEqual([150, 50, 200]); // the empty child is not a fill
});

test("standings rank by cost vs VWAP, with VWAP last and final costs only when finished", () => {
  const t = makeTrace();
  const at2 = standings(t, 2);
  expect(at2.map((r) => r.name)).toEqual(["efnn_s0", "mlp", "vwap"]);
  expect(at2[0]!.vsReference).toBe(-1);
  expect(at2[0]!.finalCost).toBeNull();
  const end = standings(t, 4);
  expect(end.map((r) => r.name)).toEqual(["efnn_s0", "mlp", "vwap"]);
  expect(end.find((r) => r.name === "mlp")!.vsReference).toBe(1);
  expect(end.every((r) => r.finished && r.done === 1)).toBe(true);
  expect(standings(t, 0).every((r) => r.costSoFar === 0 && r.done === 0)).toBe(true);
});

test("the decision card reads the step just executed", () => {
  const t = makeTrace();
  expect(decisionAt(t.policies[1]!, 2)).toMatchObject({ index: 1, q: 50, pace: 0.5 });
  expect(decisionAt(t.policies[1]!, 0).index).toBe(0);
});

test("tickets round-trip through the URL and validate like the backend", () => {
  const t = { ...defaultTicket(META), start: "14:15", horizon: 60 };
  expect(fromSearch(toSearch(t))).toEqual(t);
  expect(validate(t, META)).toEqual({});
  expect(latestStart(META, 120)).toBe("13:15");
  expect(validate({ ...t, start: "14:16" }, META).start).toMatch(/09:20 and 14:15/);
  expect(validate({ ...t, start: "09:19" }, META).start).toBeDefined();
  expect(validate({ ...t, date: "2026-09-08", sim: "tick" }, META).sim).toMatch(/Tick replay/);
  expect(validate({ ...t, policies: [] }, META).policies).toBeDefined();
  expect(fromSearch(new URLSearchParams("sec=X&date=2026-01-01"))).toBeNull();
});

test("the replay clock advances at 2 decisions/s × speed, stops at the end and restarts from it", () => {
  const r = useReplay.getState();
  r.load(60, "efnn_s0", true);
  r.advance(1);
  expect(useReplay.getState().k).toBe(2);
  useReplay.getState().setSpeed(16);
  useReplay.getState().advance(10);
  expect(useReplay.getState()).toMatchObject({ k: 60, playing: false });
  useReplay.getState().toggle();
  expect(useReplay.getState()).toMatchObject({ k: 0, playing: true });
  useReplay.getState().seek(99);
  expect(useReplay.getState()).toMatchObject({ k: 60, playing: false });
  useReplay.getState().step(-100);
  expect(useReplay.getState().k).toBe(0);
  useReplay.getState().load(30, null, false); // reduced motion: start at the finished state
  expect(useReplay.getState()).toMatchObject({ k: 30, playing: false });
});

test("labels split into a short name and its qualifier", async () => {
  const { splitLabel } = await import("@/features/execute/model");
  expect(splitLabel("PPO-MLP (5-seed ensemble)")).toEqual(["PPO-MLP", "5-seed ensemble"]);
  expect(splitLabel("PPO-eFNN + limit orders (seed 0)")).toEqual(["PPO-eFNN + limit orders", "seed 0"]);
  expect(splitLabel("VWAP")).toEqual(["VWAP", null]);
  expect(splitLabel("VWAP, always passive")).toEqual(["VWAP, always passive", null]);
});
