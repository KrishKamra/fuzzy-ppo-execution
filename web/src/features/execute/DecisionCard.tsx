import type { OrderTrace } from "@/api/client";
import { PACE_SCALE, seriesColor } from "@/lib/palette";
import { useReplay } from "@/store/replay";
import { useUi } from "@/store/ui";

import { cursorIndex, decisionAt } from "./model";

const INPUTS = [
  { label: "Order-book imbalance", lo: "against us", hi: "with us" },
  { label: "Spread variance", lo: "calm", hi: "unstable" },
  { label: "Volatility", lo: "calm", hi: "volatile" },
];

function paceWords(pace: number): string {
  if (pace < 0.95) return "slower than the VWAP schedule";
  if (pace > 1.05) return "faster than the VWAP schedule";
  return "on the VWAP schedule";
}

/** What the focus policy just did, and the three regime inputs it saw (the fuzzy brain grows from here, M6e). */
export function DecisionCard({ trace }: { trace: OrderTrace }) {
  const k = useReplay((s) => s.k);
  const focusName = useReplay((s) => s.focus);
  const theme = useUi((s) => s.effectiveTheme());
  const focus = trace.policies.find((p) => p.name === focusName) ?? trace.policies[0];
  if (!focus) return null;
  const d = decisionAt(focus, k);
  const scale = PACE_SCALE[theme];
  const paceColor = d.pace < 0.95 ? scale.slow : d.pace > 1.05 ? scale.fast : scale.neutral;
  const time = trace.market.time[cursorIndex(trace, Math.max(k, 1))] ?? trace.order.start;
  const limit = focus.steps.passive.some((x) => x > 0);
  return (
    <div className="space-y-5">
      <div className="flex items-center gap-2 text-sm">
        <span aria-hidden className="size-2 rounded-full" style={{ background: seriesColor(focus.name, theme) }} />
        <span className="font-medium">{focus.label}</span>
        <span className="tabular ml-auto font-mono text-xs text-fg-subtle">{k === 0 ? "next decision" : time}</span>
      </div>
      <div>
        <p className="text-xs text-fg-subtle">Pace</p>
        <p className="tabular mt-1 font-mono text-3xl font-semibold" style={{ color: k ? undefined : "var(--fg-subtle)" }}>
          {d.pace.toFixed(2)}
          <span className="ml-1 text-base text-fg-subtle">×</span>
        </p>
        <p className="mt-1 flex items-center gap-2 text-xs text-fg-muted">
          <span aria-hidden className="h-1.5 w-6 rounded-full" style={{ background: paceColor }} />
          {paceWords(d.pace)}
        </p>
      </div>
      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <dt className="text-xs text-fg-subtle">Child order</dt>
          <dd className="tabular mt-0.5 font-mono">{Math.round(d.q).toLocaleString()} sh</dd>
        </div>
        <div>
          <dt className="text-xs text-fg-subtle">Posted passively</dt>
          <dd className="tabular mt-0.5 font-mono">{limit ? `${Math.round(d.passive * 100)}%` : "–"}</dd>
        </div>
      </dl>
      <div className="space-y-3">
        <p className="text-xs text-fg-subtle">What it saw (train-standardised)</p>
        {INPUTS.map((inp, j) => {
          const v = d.regime[j] ?? 0;
          const pct = Math.max(-1, Math.min(1, v / 3)) * 50; // ±3 standard deviations fill the half-bar
          return (
            <div key={inp.label}>
              <div className="flex justify-between text-xs">
                <span className="text-fg-muted">{inp.label}</span>
                <span className="tabular font-mono text-fg-subtle">{v >= 0 ? "+" : "−"}{Math.abs(v).toFixed(2)}σ</span>
              </div>
              <div className="relative mt-1.5 h-1.5 rounded-full bg-surface-3" aria-hidden>
                <span className="absolute top-0 left-1/2 h-full w-px bg-[var(--hairline-strong)]" />
                <span
                  className="bg-aurora absolute top-0 h-full rounded-full"
                  style={pct >= 0 ? { left: "50%", width: `${pct}%` } : { left: `${50 + pct}%`, width: `${-pct}%` }}
                />
              </div>
              <div className="mt-1 flex justify-between text-[0.65rem] text-fg-subtle">
                <span>{inp.lo}</span>
                <span>{inp.hi}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
