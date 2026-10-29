import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";

import { makeQueryClient, routes } from "@/app/App";
import { FuzzyBrain } from "@/features/brain/FuzzyBrain";
import { activeRule, gaussian, paceColor, paceVerdict, ruleSentence } from "@/features/brain/fuzzy";
import { useUi } from "@/store/ui";

import { makeTrace, META, RULES } from "./fixtures";

vi.mock("@/components/charts/EChart", () => ({ EChart: ({ label }: { label: string }) => <div role="img" aria-label={label} /> }));

function mockApi() {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.endsWith("/health")) return new Response(JSON.stringify({ status: "ok" }));
    if (url.endsWith("/meta")) return new Response(JSON.stringify(META));
    if (url.includes("/rules")) return new Response(JSON.stringify(RULES));
    if (url.includes("/order/random")) return new Response(JSON.stringify({ security: "ADE IS Equity", date: "2026-09-29", start: "11:00", horizon: 30, side: "buy", participation: 0.05 }));
    if (url.endsWith("/order")) return new Response(JSON.stringify(makeTrace()));
    return new Response("{}", { status: 404 });
  })); // fmt: skip
}

function renderAt(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(<QueryClientProvider client={makeQueryClient()}><RouterProvider router={router} /></QueryClientProvider>);
  return router;
}

beforeEach(() => {
  localStorage.clear();
  useUi.setState({ theme: "dark", presentation: false, paletteOpen: false });
  vi.stubGlobal("matchMedia", (q: string) => ({ matches: q.includes("reduce"), addEventListener() {}, removeEventListener() {} }));
  mockApi();
});
afterEach(() => vi.unstubAllGlobals());

test("fuzzy helpers: membership, the active rule and rules as sentences", () => {
  expect(gaussian(0, 0, 1)).toBe(1);
  expect(gaussian(1, 0, 1)).toBeCloseTo(Math.exp(-0.5));
  expect(activeRule([0.2, 0.7, 0.1])).toBe(1);
  expect(activeRule(null)).toBeNull();
  expect(paceVerdict(0.9)).toBe("slow down");
  expect(paceVerdict(1)).toBe("keep to schedule");
  expect(paceVerdict(1.1)).toBe("speed up");
  const s = ruleSentence(RULES.rules[2]!);
  expect(s.clauses.map((c) => `${c.input}:${c.term}`)).toEqual(["Imbalance:neutral", "Spread variance:high", "Volatility:high"]);
  expect(s.then).toBe("pace 0.88× (slow down)");
  expect(ruleSentence({ ...RULES.rules[0]!, passive: 0.91 }).then).toContain("post 91% passively");
  expect(paceColor(1, "dark")).toBe("rgb(74, 76, 87)"); // on schedule: the neutral midpoint
  expect(paceColor(0.5, "dark")).toBe("rgb(57, 135, 229)"); // clamps to the slow pole
});

test("the fuzzy brain states the active rule and its firing", () => {
  render(<FuzzyBrain rules={RULES} phi={[0.7, 0.2, 0.1]} regime={[0.5, -1, 2]} />);
  expect(screen.getByText("Active rule R0")).toBeInTheDocument();
  expect(screen.getByText("fires 70%")).toBeInTheDocument();
  expect(screen.getByText(/pace 0.97× \(keep to schedule\)/)).toBeInTheDocument();
  const bars = within(screen.getByRole("list", { name: "Rule firing strengths" })).getAllByRole("listitem");
  expect(bars.map((b) => b.textContent)).toEqual(["R070%", "R120%", "R210%"]);
});

test("Execute shows the fuzzy brain for a fuzzy policy and explains why an MLP has none", async () => {
  const user = userEvent.setup();
  renderAt("/execute?sec=ADE+IS+Equity&date=2026-09-29&start=11%3A00&h=30&side=buy&p=0.05&sim=bar&pol=efnn_s0%2Cmlp%2Cvwap");
  expect(await screen.findByText("Active rule R0")).toBeInTheDocument();
  await user.click(within(screen.getByRole("list", { name: "Legend" })).getByRole("button", { name: /PPO-MLP/ }));
  expect(await screen.findByText(/no single rule base to show/)).toBeInTheDocument();
  expect(screen.queryByText("Active rule R0")).not.toBeInTheDocument();
});

test("Brain draws the regime space (2D without WebGL), the reasoning and the rule table", async () => {
  const router = renderAt("/brain");
  await waitFor(() => expect(router.state.location.pathname).toBe("/brain"));
  expect(await screen.findByLabelText("2D projections of the regime space")).toBeInTheDocument(); // jsdom has no WebGL
  expect(await screen.findByText("Active rule R0")).toBeInTheDocument();
  const table = screen.getByRole("table");
  expect(within(table).getAllByRole("row")).toHaveLength(4); // header + 3 rules
  expect(within(table).getByText("pace 0.88× (slow down)")).toBeInTheDocument();
});
