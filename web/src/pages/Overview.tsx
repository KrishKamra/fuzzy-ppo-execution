import { ArrowRight, BookOpenCheck, Repeat2, Waypoints } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router";

import { Eyebrow, Panel } from "@/components/ui/primitives";

/** Headline figures, exactly as in docs/RESULTS.md (bps of arrival notional; negative = cheaper than VWAP). */
const CLAIMS: { icon: ReactNode; value: string; unit: string; label: string; detail: string }[] = [
  {
    icon: <Repeat2 className="size-4" />,
    value: "−0.16 to −0.24",
    unit: "basis points vs VWAP",
    label: "Beats VWAP out of sample",
    detail: "Held-out October test sessions, 5-seed ensembles; reproduced by an independent retraining.",
  },
  {
    icon: <Waypoints className="size-4" />,
    value: "−0.43",
    unit: "basis points vs the always-passive rule",
    label: "Learns when to post limit orders",
    detail: "Beats always-posting-passively with a lighter tail, replayed against the real order book.",
  },
  {
    icon: <BookOpenCheck className="size-4" />,
    value: "5 / 5",
    unit: "eFNN seeds follow execution theory",
    label: "Rules you can read",
    detail: "Calm regimes trade at or ahead of schedule; stressed regimes slow down, as execution theory predicts.",
  },
];

export function Overview() {
  return (
    <div className="mx-auto max-w-5xl">
      <section className="pt-10 pb-14 text-center">
        <Eyebrow>Intelligent systems · Optimal execution on NSE</Eyebrow>
        <h1 className="mx-auto mt-5 max-w-3xl text-5xl leading-[1.05] font-semibold tracking-tight text-balance sm:text-6xl">
          An execution agent <span className="text-aurora">you can read.</span>
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-lg text-pretty text-fg-muted">
          A PPO agent slices large orders through the trading day. An evolving fuzzy network turns order-book
          imbalance, spread variance and volatility into regimes, and every decision traces back to a rule in plain
          language.
        </p>
        <div className="mt-9 flex flex-wrap justify-center gap-3">
          <Link
            to="/execute?random=1"
            className="bg-aurora inline-flex h-11 items-center gap-2 rounded-[var(--radius-control)] px-5 font-medium text-white shadow-[0_12px_32px_-14px_var(--accent-2)] transition hover:brightness-110"
          >
            Run a live order <ArrowRight aria-hidden className="size-4" />
          </Link>
          <Link
            to="/evidence"
            className="inline-flex h-11 items-center rounded-[var(--radius-control)] border border-hairline bg-surface-1 px-5 font-medium text-fg transition hover:bg-surface-2"
          >
            See the evidence
          </Link>
        </div>
      </section>
      <section aria-label="Headline results" className="grid gap-4 md:grid-cols-3">
        {CLAIMS.map((c) => (
          <Panel key={c.label} className="p-6">
            <div className="flex items-center gap-2 text-sm text-fg-muted">
              <span className="text-accent-2">{c.icon}</span>
              {c.label}
            </div>
            <p className="tabular mt-4 font-mono text-3xl font-semibold tracking-tight">{c.value}</p>
            <p className="mt-1 text-xs text-fg-subtle">{c.unit}</p>
            <p className="mt-3 text-sm leading-relaxed text-fg-muted">{c.detail}</p>
          </Panel>
        ))}
      </section>
      <p className="mt-6 text-center text-xs text-fg-subtle">
        Costs in basis points of the order's value, in a market simulator calibrated to Bloomberg data for three
        Nifty-50 stocks. Not a claim about live trading.
      </p>
    </div>
  );
}
