import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";

import { api, ApiError } from "@/api/client";
import { makeQueryClient, routes } from "@/app/App";
import { useUi } from "@/store/ui";

const META = {
  securities: ["ADE IS Equity"],
  sessions: [
    { security: "ADE IS Equity", date: "2026-09-29", split: "val", ticks: true },
    { security: "ADE IS Equity", date: "2026-10-01", split: "test", ticks: true },
  ],
  horizons: [30, 60, 120],
  earliest_start: "09:20",
  latest_end: "15:15",
  participation: [0.01, 0.2],
  policies: [
    { name: "efnn_s0", label: "eFNN", kind: "ppo", description: "", explains: true, limit_orders: false },
    { name: "vwap", label: "VWAP", kind: "baseline", description: "", explains: false, limit_orders: false },
  ],
  default_policies: ["efnn_s0", "vwap"],
  runs: ["m4_efnn_s0"],
  regime_inputs: ["a", "b", "c"],
};

function mockApi(online = true) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (!online) throw new TypeError("Failed to fetch");
      const body = url.endsWith("/health") ? { status: "ok" } : url.endsWith("/meta") ? META : {};
      return new Response(JSON.stringify(body), { status: 200 });
    }),
  );
}

function renderAt(path = "/") {
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
  mockApi();
});
afterEach(() => vi.unstubAllGlobals());

test("the overview renders the headline and the API status", async () => {
  renderAt("/");
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("An execution agent you can read.");
  expect(await screen.findByText("API live")).toBeInTheDocument();
});

test("number keys switch screens, but not while typing", async () => {
  const user = userEvent.setup();
  const router = renderAt("/");
  await user.keyboard("2");
  // Execute is a lazy route: the location commits once its chunk has loaded.
  await waitFor(() => expect(router.state.location.pathname).toBe("/execute"));
  expect(await screen.findByText("Pick an order and press Run")).toBeInTheDocument();
  await user.keyboard("{Control>}k{/Control}");
  const input = await screen.findByPlaceholderText(/Go to a screen/);
  await user.type(input, "3");
  expect(router.state.location.pathname).toBe("/execute"); // typing in the palette never navigates
});

test("the palette navigates and closes", async () => {
  const user = userEvent.setup();
  const router = renderAt("/");
  await user.click(screen.getByRole("button", { name: /Search/ }));
  const dialog = await screen.findByRole("dialog");
  await user.click(within(dialog).getByText("Evidence"));
  await waitFor(() => expect(router.state.location.pathname).toBe("/evidence")); // a lazy route
  await waitFor(() => expect(screen.queryByPlaceholderText(/Go to a screen/)).not.toBeInTheDocument());
});

test("theme and presentation mode reach <html> and persist", async () => {
  const user = userEvent.setup();
  renderAt("/");
  await user.keyboard("t");
  expect(document.documentElement.dataset.theme).toBe("light");
  await user.keyboard("t");
  expect(document.documentElement.dataset.theme).toBe("dark");
  await user.keyboard("p");
  expect(document.documentElement.dataset.theme).toBe("light"); // presentation forces light for projectors
  expect(document.documentElement).toHaveAttribute("data-presentation");
  expect(JSON.parse(localStorage.getItem("hftf.ui") ?? "{}")).toEqual({ theme: "dark", presentation: true });
});

test("an offline API is reported, not crashed on", async () => {
  mockApi(false);
  renderAt("/execute");
  expect(await screen.findByText("API offline")).toBeInTheDocument();
  // useMeta retries once (1 s back-off) before reporting, so wait past the default 1 s.
  expect(await screen.findByRole("alert", {}, { timeout: 4000 })).toHaveTextContent("hftf serve");
});

test("the client maps API errors to readable messages", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify({ detail: "no val/test session for X on 2030-01-01" }), { status: 400 })),
  );
  await expect(api.meta()).rejects.toEqual(new ApiError(400, "no val/test session for X on 2030-01-01"));
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: [{ msg: "bad" }] }), { status: 422 })));
  await expect(api.meta()).rejects.toMatchObject({ status: 422, message: "The order ticket is invalid." });
});
