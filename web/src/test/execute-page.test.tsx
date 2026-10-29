import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";

import { makeQueryClient, routes } from "@/app/App";
import { useReplay } from "@/store/replay";
import { useUi } from "@/store/ui";

import { makeTrace, META } from "./fixtures";

// ECharts needs a real canvas; the stub exposes what each chart would draw.
vi.mock("@/components/charts/EChart", () => ({
  EChart: ({ label, option }: { label: string; option: { series?: { id: string }[] } }) => (
    <div role="img" aria-label={label} data-series={(option.series ?? []).map((s) => s.id).join(",")} />
  ),
}));

const calls: { url: string; body?: unknown }[] = [];

function mockApi() {
  calls.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, body: init?.body ? JSON.parse(String(init.body)) : undefined });
      if (url.endsWith("/health")) return new Response(JSON.stringify({ status: "ok" }));
      if (url.endsWith("/meta")) return new Response(JSON.stringify(META));
      if (url.includes("/order/random"))
        return new Response(JSON.stringify({ security: "APHS IS Equity", date: "2026-09-29", start: "12:42", horizon: 60, side: "sell", participation: 0.1 }));
      if (url.endsWith("/order")) return new Response(JSON.stringify(makeTrace()));
      return new Response("{}", { status: 404 });
    }),
  );
}

function renderAt(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(
    <QueryClientProvider client={makeQueryClient()}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return router;
}

beforeEach(() => {
  localStorage.clear();
  useUi.setState({ theme: "dark", presentation: false, paletteOpen: false });
  // Reduced motion: the replay starts at the finished state, so the assertions do not race the clock.
  vi.stubGlobal("matchMedia", (q: string) => ({ matches: q.includes("reduce"), addEventListener() {}, removeEventListener() {} }));
  mockApi();
});
afterEach(() => vi.unstubAllGlobals());

test("running an order requests every selected policy plus VWAP, and shows charts and standings", async () => {
  const user = userEvent.setup();
  const router = renderAt("/execute");
  const form = await screen.findByRole("form", { name: "Order ticket" });
  await user.click(within(form).getByRole("button", { name: /PPO-MLP/ })); // deselect MLP
  await user.click(within(form).getByRole("button", { name: /Run order/ }));
  await waitFor(() => expect(calls.some((c) => c.url.endsWith("/api/order"))).toBe(true));
  const body = calls.find((c) => c.url.endsWith("/api/order"))!.body as { policies: string[] };
  expect(body.policies).toEqual(["efnn_s0", "vwap"]); // VWAP stays as the reference
  expect(router.state.location.search).toContain("pol=efnn_s0%2Cvwap");
  expect(await screen.findByRole("heading", { level: 1 })).toHaveTextContent("Buy 400 ADE · 11:00–11:04");
  const table = screen.getByRole("table");
  const rows = within(table).getAllByRole("row").slice(1);
  // Each row header is the policy (its full name in the title) plus a "done · passive" detail line.
  expect(rows.map((r) => within(r).getByTitle(/./).getAttribute("title"))).toEqual(["PPO-eFNN (seed 0)", "PPO-MLP (5-seed ensemble)", "VWAP"]);
  expect(rows[0]).toHaveTextContent("100% done");
  expect(within(rows[0]!).getByText("−1.00")).toBeInTheDocument(); // final cost 4 vs VWAP 5
  expect(screen.getByRole("img", { name: /Running cost minus VWAP/ })).toHaveAttribute("data-series", "efnn_s0,mlp,vwap");
});

test("the transport steps with the keyboard and seeks with the scrubber", async () => {
  const user = userEvent.setup();
  renderAt("/execute?sec=ADE+IS+Equity&date=2026-09-29&start=11%3A00&h=30&side=buy&p=0.05&sim=bar&pol=efnn_s0%2Cvwap");
  const scrubber = await screen.findByRole("slider", { name: /Replay position/ });
  expect(scrubber).toHaveValue("4"); // reduced motion: finished
  await user.keyboard("{ArrowLeft}{ArrowLeft}");
  expect(useReplay.getState().k).toBe(2);
  expect(scrubber).toHaveAttribute("aria-valuetext", expect.stringContaining("decision 2 of 4"));
  await user.click(screen.getByRole("button", { name: /Play/ }));
  expect(useReplay.getState().playing).toBe(true);
  act(() => useReplay.getState().seek(1));
  expect(scrubber).toHaveValue("1");
});

test("?random=1 draws a ticket and runs it", async () => {
  const router = renderAt("/execute?random=1");
  await waitFor(() => expect(router.state.location.search).toContain("sec=APHS+IS+Equity"));
  expect(router.state.location.search).not.toContain("random");
  // The heading changes from the empty state to the order once the (mocked) trace returns.
  expect(await screen.findByRole("heading", { level: 1, name: /ADE · 11:00/ })).toBeInTheDocument();
});

test("an invalid start disables Run and explains why", async () => {
  const user = userEvent.setup();
  renderAt("/execute");
  const start = await screen.findByLabelText("Start (IST)");
  await user.clear(start);
  await user.type(start, "15:00");
  expect(await screen.findByText(/Start between 09:20 and 14:15/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Run order/ })).toBeDisabled();
});
