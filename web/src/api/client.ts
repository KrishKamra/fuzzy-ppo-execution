/**
 * Typed client for the demo API (apps/api). Types come from `schema.d.ts`, generated from the backend's OpenAPI
 * schema (`npm run gen:api`), so a contract change fails the type check instead of the demo.
 */
import type { components } from "./schema";

type S = components["schemas"];
export type Meta = S["Meta"];
export type OrderTicket = S["OrderTicket"];
export type OrderRequest = S["OrderRequest"];
export type OrderTrace = S["OrderTrace"];
export type PolicyTrace = S["PolicyTrace"];
export type Rules = S["Rules"];
export type RunCurves = S["RunCurves"];
export type Results = S["Results"];

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  } catch {
    throw new ApiError(0, "The API is not reachable. Start it with `uv run hftf serve`.");
  }
  if (!res.ok) {
    // FastAPI errors carry {"detail": "..."} (or a list of validation errors).
    const body = (await res.json().catch(() => null)) as { detail?: unknown } | null;
    const detail = body?.detail;
    const message =
      typeof detail === "string" ? detail : Array.isArray(detail) ? "The order ticket is invalid." : res.statusText;
    throw new ApiError(res.status, message);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  meta: () => request<Meta>("/meta"),
  randomTicket: (split = "val", sim: "bar" | "tick" = "bar", seed = 0) =>
    request<OrderTicket>(`/order/random?${new URLSearchParams({ split, sim, seed: String(seed) })}`),
  order: (req: OrderRequest) => request<OrderTrace>("/order", { method: "POST", body: JSON.stringify(req) }),
  rules: (policy: string) => request<Rules>(`/policies/${encodeURIComponent(policy)}/rules`),
  run: (name: string) => request<RunCurves>(`/runs/${encodeURIComponent(name)}`),
  results: () => request<Results>("/results"),
};
