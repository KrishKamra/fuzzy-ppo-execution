import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router";

import type { Results, RunCurves } from "@/api/client";
import { makeQueryClient, routes } from "@/app/App";
import { summarize } from "@/features/training/charts";
import { runLabel } from "@/pages/Training";
import { useUi } from "@/store/ui";

import { META } from "./fixtures";

vi.mock("@/components/charts/EChart", () => ({ EChart: ({ label }: { label: string }) => <div role="img" aria-label={label} /> }));

const stat = (policy: string, reference: string, mean: number, extra = {}) => ({ policy, reference, mean, lo: mean - 0.1, hi: mean + 0.1, date_lo: mean - 0.12, date_hi: mean + 0.12, holm_p: 0.004, ...extra });
const RESULTS: Results = {
  tables: [
    { id: "test_vs_vwap", title: "", columns: [], rows: [stat("mlp", "vwap", -0.16), stat("anfis", "vwap", -0.24), stat("efnn", "vwap", -0.22)] },
    { id: "test_vs_mlp", title: "", columns: [], rows: [stat("anfis", "mlp", -0.08), stat("efnn", "mlp", -0.06)] },
    { id: "bar_vs_vwap", title: "", columns: [], rows: [stat("efnn", "vwap", -0.12)] },
    { id: "tick_vs_vwap", title: "", columns: [], rows: [stat("efnn", "vwap", -0.18)] },
    { id: "limit_vs_vwap", title: "", columns: [], rows: [stat("lip_efnn", "vwap", -2.36)] },
    { id: "limit_vs_passive", title: "", columns: [], rows: [stat("lip_efnn", "vwap_passive100", -0.46)] },
    { id: "robust_gamma", title: "", columns: [], rows: [stat("mlp", "vwap", -0.1, { gamma: 0.3 }), stat("efnn", "vwap", -0.2, { gamma: 0.3 })] },
  ],
}; // fmt: skip

const RUN: RunCurves = {
  run: "m4_efnn_s0",
  progress: { iteration: [1, 2, 3, 4], val_vs_vwap: [0.1, null, -0.25, -0.2], val_vs_vwap_se: [0.05, null, 0.05, 0.05], train_vs_vwap: [0, null, -0.2, -0.2], kl: [0.003, 0.004, 0.003, 0.002], n_rules: [4, 5, 5, 4], edit_kl: [0.001, 0, 0, 0.002] },
  events: [
    { iteration: 1, kind: "add", rules: "4", n_rules: 5, accepted: 1, kl: 0.001, detail: "{}" },
    { iteration: 4, kind: "prune", rules: "2", n_rules: 4, accepted: 1, kl: 0.004, detail: "{}" },
    { iteration: 3, kind: "merge", rules: "1 2", n_rules: 4, accepted: 0, kl: 0.03, detail: "{}" },
  ],
}; // fmt: skip

function renderAt(path: string, runs = ["m4_efnn_s0", "m4_mlp_s0"]) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.endsWith("/health")) return new Response(JSON.stringify({ status: "ok" }));
    if (url.endsWith("/meta")) return new Response(JSON.stringify({ ...META, runs }));
    if (url.endsWith("/results")) return new Response(JSON.stringify(RESULTS));
    if (url.endsWith("/runs/m4_efnn_s0")) return new Response(JSON.stringify(RUN));
    if (url.endsWith("/runs/m4_mlp_s0")) return new Response(JSON.stringify({ run: "m4_mlp_s0", progress: { iteration: [1, 2], val_vs_vwap: [0, -0.1], val_vs_vwap_se: [0.05, 0.05], kl: [0.003, 0.002] }, events: [] }));
    return new Response("{}", { status: 404 });
  })); // fmt: skip
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(<QueryClientProvider client={makeQueryClient()}><RouterProvider router={router} /></QueryClientProvider>);
  return router;
}

beforeEach(() => {
  localStorage.clear();
  useUi.setState({ theme: "dark", presentation: false, paletteOpen: false });
});
afterEach(() => vi.unstubAllGlobals());

test("training summaries: best checkpoint, accepted and undone edits, largest accepted edit KL", () => {
  expect(summarize(RUN)).toEqual({ bestIteration: 3, bestEdge: -0.25, finalRules: 4, edits: { add: 1, merge: 0, prune: 1 }, rejected: 1, maxEditKl: 0.004 });
  expect(runLabel("m4_efnn_s0")).toBe("eFNN · seed 0");
  expect(runLabel("m5_lip_efnn_s2")).toBe("eFNN + limit orders · seed 2");
  expect(runLabel("custom")).toBe("custom");
});

test("Evidence lists the claims, honest about what is not established, and toggles the replication", async () => {
  const user = userEvent.setup();
  renderAt("/evidence");
  expect(await screen.findByText("The fuzzy architecture beats the neural network")).toBeInTheDocument();
  expect(screen.getByText("Not established")).toBeInTheDocument();
  const section = (await screen.findByText(/in two independent runs/)).closest("section, div[id]") as HTMLElement;
  const rows = () => within(section).getAllByText(/seeds$/).map((x) => x.textContent);
  expect(rows()).toEqual(["ours · 5 seeds", "replication · 3 seeds", "ours · 5 seeds", "replication · 3 seeds", "ours · 5 seeds", "replication · 3 seeds"]);
  await user.click(within(section).getByRole("radio", { name: "Our run" }));
  expect(rows()).toEqual(["ours · 5 seeds", "ours · 5 seeds", "ours · 5 seeds"]);
  expect(screen.getByText(/the date Holm p of 0.016 is the floor for 8 dates/)).toBeInTheDocument();
});

test("Training shows the selected checkpoint and edits, and says an MLP has no rule base", async () => {
  const user = userEvent.setup();
  const router = renderAt("/training");
  await waitFor(() => expect(router.state.location.pathname).toBe("/training"));
  expect(await screen.findByText("iter 3")).toBeInTheDocument();
  expect(screen.getByText("2 / 1")).toBeInTheDocument(); // 2 accepted, 1 undone by the KL guard
  expect(screen.getByRole("img", { name: /fuzzy rules over training/ })).toBeInTheDocument();
  await user.click(screen.getByRole("radio", { name: "MLP · seed 0" }));
  expect(await screen.findByText("An MLP has no rule base")).toBeInTheDocument();
  expect(screen.getByText("no rules (MLP)")).toBeInTheDocument();
});
