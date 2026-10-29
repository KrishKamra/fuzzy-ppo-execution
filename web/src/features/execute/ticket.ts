/**
 * The order ticket: form state, validation (the same window the backend enforces) and URL encoding, so every
 * replay is a shareable link (/execute?sec=…&date=…).
 */
import type { Meta, OrderRequest } from "@/api/client";

export interface Ticket {
  security: string;
  date: string;
  start: string; // IST HH:MM, first decision
  horizon: number;
  side: "buy" | "sell";
  participation: number; // Q / expected volume over the horizon, 0.01 … 0.2
  sim: "bar" | "tick";
  policies: string[];
}

export const MAX_POLICIES = 6; // the validated categorical palette has six slots

export const toMinutes = (hhmm: string) => {
  const [h, m] = hhmm.split(":").map(Number);
  return (h ?? 0) * 60 + (m ?? 0);
};
export const toClock = (minutes: number) =>
  `${String(Math.floor(minutes / 60)).padStart(2, "0")}:${String(minutes % 60).padStart(2, "0")}`;

/** Latest valid start for a horizon: the order must finish by the end of the grid. */
export function latestStart(meta: Pick<Meta, "latest_end">, horizon: number): string {
  return toClock(toMinutes(meta.latest_end) - horizon);
}

export function sessionOf(meta: Meta, security: string, date: string) {
  return meta.sessions.find((s) => s.security === security && s.date === date);
}

export function defaultTicket(meta: Meta): Ticket {
  const val = meta.sessions.filter((s) => s.split === "val");
  const s = val.find((x) => x.ticks) ?? val[0] ?? meta.sessions[0];
  return {
    security: s?.security ?? meta.securities[0] ?? "",
    date: s?.date ?? "",
    start: "11:00",
    horizon: 60,
    side: "buy",
    participation: 0.05,
    sim: "bar",
    policies: meta.default_policies.slice(0, MAX_POLICIES),
  };
}

/** Problems with a ticket, keyed by field ("" = none). */
export function validate(t: Ticket, meta: Meta): Partial<Record<keyof Ticket, string>> {
  const errors: Partial<Record<keyof Ticket, string>> = {};
  const session = sessionOf(meta, t.security, t.date);
  if (!session) errors.date = "No session for this stock on this date";
  const start = toMinutes(t.start);
  if (!/^\d{2}:\d{2}$/.test(t.start) || start < toMinutes(meta.earliest_start) || start > toMinutes(latestStart(meta, t.horizon)))
    errors.start = `Start between ${meta.earliest_start} and ${latestStart(meta, t.horizon)} for a ${t.horizon}-min order`;
  if (!meta.horizons.includes(t.horizon)) errors.horizon = `Horizon must be ${meta.horizons.join(", ")} min`;
  if (!(t.participation > 0 && t.participation <= 0.3)) errors.participation = "Size between 1% and 30%";
  if (t.sim === "tick" && session && !session.ticks) errors.sim = "Tick replay exists only for val Sep 21-30 and test";
  if (t.policies.length === 0) errors.policies = "Pick at least one policy";
  if (t.policies.length > MAX_POLICIES) errors.policies = `At most ${MAX_POLICIES} policies`;
  return errors;
}

export function toRequest(t: Ticket): OrderRequest {
  return { ...t };
}

const KEYS = { security: "sec", date: "date", start: "start", horizon: "h", side: "side", participation: "p", sim: "sim", policies: "pol" } as const; // fmt: skip

export function toSearch(t: Ticket): URLSearchParams {
  return new URLSearchParams({
    [KEYS.security]: t.security, [KEYS.date]: t.date, [KEYS.start]: t.start, [KEYS.horizon]: String(t.horizon),
    [KEYS.side]: t.side, [KEYS.participation]: String(t.participation), [KEYS.sim]: t.sim,
    [KEYS.policies]: t.policies.join(","),
  }); // fmt: skip
}

/** A ticket from the URL, or null if the URL does not carry a complete one. */
export function fromSearch(q: URLSearchParams): Ticket | null {
  const get = (k: string) => q.get(k);
  const security = get(KEYS.security), date = get(KEYS.date), start = get(KEYS.start); // fmt: skip
  const horizon = Number(get(KEYS.horizon)), participation = Number(get(KEYS.participation)); // fmt: skip
  const side = get(KEYS.side), sim = get(KEYS.sim) ?? "bar"; // fmt: skip
  if (!security || !date || !start || !Number.isFinite(horizon) || !Number.isFinite(participation)) return null;
  if (side !== "buy" && side !== "sell") return null;
  return {
    security, date, start, horizon, side, participation,
    sim: sim === "tick" ? "tick" : "bar",
    policies: (get(KEYS.policies) ?? "").split(",").filter(Boolean),
  }; // fmt: skip
}
