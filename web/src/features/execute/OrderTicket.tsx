import { clsx } from "clsx";
import { ArrowDownRight, ArrowUpRight, Loader2, Play, Shuffle } from "lucide-react";
import { useMemo } from "react";

import type { Meta } from "@/api/client";
import { Button, Field, Segmented } from "@/components/ui/primitives";
import { seriesColor } from "@/lib/palette";
import { useUi } from "@/store/ui";

import { REFERENCE } from "./model";
import { latestStart, MAX_POLICIES, sessionOf, type Ticket, validate } from "./ticket";

const CONTROL =
  "h-9 w-full rounded-[var(--radius-control)] border border-hairline bg-surface-2 px-2.5 text-sm text-fg " +
  "outline-none focus-visible:border-accent-2";

interface Props {
  meta: Meta;
  ticket: Ticket;
  onChange: (t: Ticket) => void;
  onRun: () => void;
  onRandom: () => void;
  busy: boolean;
  showTest: boolean;
  onShowTest: (v: boolean) => void;
}

export function OrderTicket({ meta, ticket, onChange, onRun, onRandom, busy, showTest, onShowTest }: Props) {
  const theme = useUi((s) => s.effectiveTheme());
  const errors = validate(ticket, meta);
  const set = (patch: Partial<Ticket>) => onChange({ ...ticket, ...patch });
  const dates = useMemo(
    () => meta.sessions.filter((s) => s.security === ticket.security && (showTest || s.split !== "test")),
    [meta, ticket.security, showTest],
  );
  const session = sessionOf(meta, ticket.security, ticket.date);
  const ppo = meta.policies.filter((p) => p.kind === "ppo");
  const baselines = meta.policies.filter((p) => p.kind === "baseline" && p.name !== REFERENCE);

  const togglePolicy = (name: string) => {
    const on = ticket.policies.includes(name);
    if (!on && ticket.policies.length >= MAX_POLICIES) return;
    set({ policies: on ? ticket.policies.filter((n) => n !== name) : [...ticket.policies, name] });
  };

  const chip = (p: Meta["policies"][number]) => {
    const on = ticket.policies.includes(p.name);
    const full = !on && ticket.policies.length >= MAX_POLICIES;
    return (
      <button
        key={p.name}
        type="button"
        aria-pressed={on}
        disabled={full}
        title={full ? `At most ${MAX_POLICIES} policies` : p.description || p.label}
        onClick={() => togglePolicy(p.name)}
        className={clsx(
          "flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs transition-colors",
          on ? "border-hairline-strong bg-surface-3 text-fg" : "border-hairline text-fg-muted hover:text-fg",
          "disabled:cursor-not-allowed disabled:opacity-40",
        )}
      >
        <span aria-hidden className="size-2 rounded-full" style={{ background: on ? seriesColor(p.name, theme) : "transparent", boxShadow: `inset 0 0 0 1.5px ${seriesColor(p.name, theme)}` }} />
        {p.label}
      </button>
    );
  };

  return (
    <form
      aria-label="Order ticket"
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault();
        if (Object.keys(errors).length === 0) onRun();
      }}
    >
      <Field label="Stock" htmlFor="t-sec">
        <select id="t-sec" className={CONTROL} value={ticket.security} onChange={(e) => {
          const security = e.target.value;
          const first = meta.sessions.find((s) => s.security === security && s.date === ticket.date) ?? meta.sessions.find((s) => s.security === security && s.split === "val");
          set({ security, date: first?.date ?? ticket.date });
        }}>
          {meta.securities.map((s) => (
            <option key={s} value={s}>{s.replace(" IS Equity", "")}</option>
          ))}
        </select>
      </Field>

      <Field label="Session" htmlFor="t-date" error={errors.date}>
        <select id="t-date" className={CONTROL} value={ticket.date} onChange={(e) => set({ date: e.target.value })}>
          {(["val", "test"] as const).map((split) => {
            const group = dates.filter((s) => s.split === split);
            return group.length ? (
              <optgroup key={split} label={split === "val" ? "Validation (Sep)" : "Test (Oct) · out of sample"}>
                {group.map((s) => (
                  <option key={s.date} value={s.date}>{s.date}{s.ticks ? "  · tick replay" : ""}</option>
                ))}
              </optgroup>
            ) : null;
          })}
        </select>
        <label className="mt-2 flex items-center gap-2 text-xs text-fg-subtle">
          <input type="checkbox" checked={showTest} onChange={(e) => onShowTest(e.target.checked)} className="accent-[var(--accent-2)]" />
          Include test sessions (held out; showcase only)
        </label>
      </Field>

      <div className="grid grid-cols-2 gap-3">
        <Field label="Start (IST)" htmlFor="t-start" error={errors.start}>
          <input id="t-start" type="time" step={60} min={meta.earliest_start} max={latestStart(meta, ticket.horizon)}
            className={clsx(CONTROL, "tabular font-mono")} value={ticket.start} onChange={(e) => set({ start: e.target.value })} />
        </Field>
        <Field label="Side">
          <Segmented label="Side" value={ticket.side} onChange={(side) => set({ side })} options={[
            { value: "buy", label: <span className="inline-flex items-center gap-1"><ArrowUpRight className="size-3.5" />Buy</span> },
            { value: "sell", label: <span className="inline-flex items-center gap-1"><ArrowDownRight className="size-3.5" />Sell</span> },
          ]} />
        </Field>
      </div>

      <Field label="Horizon" error={errors.horizon}>
        <Segmented label="Horizon" value={ticket.horizon} onChange={(horizon) => set({ horizon })}
          options={meta.horizons.map((h) => ({ value: h, label: `${h} min` }))} />
      </Field>

      <Field label={`Size: ${(ticket.participation * 100).toFixed(1)}% of the expected volume`} htmlFor="t-size" error={errors.participation}>
        <input id="t-size" type="range" min={0.01} max={0.2} step={0.005} value={ticket.participation}
          onChange={(e) => set({ participation: Number(e.target.value) })} className="w-full accent-[var(--accent-2)]" />
      </Field>

      <Field label="Simulator" error={errors.sim}>
        <Segmented label="Simulator" value={ticket.sim} onChange={(sim) => set({ sim })} options={[
          { value: "bar", label: "BarSim", title: "1-minute bars, parametric impact" },
          { value: "tick", label: "TickSim", disabled: !session?.ticks, title: session?.ticks ? "Fills against the replayed order book" : "No tick data for this session" },
        ]} />
      </Field>

      <fieldset className="space-y-2">
        <legend className="mb-1.5 text-xs font-medium text-fg-muted">
          Policies <span className="text-fg-subtle">({ticket.policies.length}/{MAX_POLICIES}; VWAP is always the reference)</span>
        </legend>
        <div className="flex flex-wrap gap-1.5">{ppo.map(chip)}</div>
        <div className="flex flex-wrap gap-1.5">{baselines.map(chip)}</div>
        {errors.policies && <p role="alert" className="text-xs text-bad">{errors.policies}</p>}
      </fieldset>

      <div className="flex gap-2 pt-1">
        <Button type="submit" variant="primary" className="flex-1 justify-center" disabled={busy || Object.keys(errors).length > 0}>
          {busy ? <Loader2 aria-hidden className="size-4 animate-spin" /> : <Play aria-hidden className="size-4" />}
          Run order
        </Button>
        <Button onClick={onRandom} disabled={busy} title="A random order from the training distribution">
          <Shuffle aria-hidden className="size-4" />
          <span className="sr-only sm:not-sr-only">Surprise me</span>
        </Button>
      </div>
    </form>
  );
}
