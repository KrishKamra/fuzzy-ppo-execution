import { clsx } from "clsx";

import type { Rules } from "@/api/client";
import type { Theme } from "@/lib/palette";
import { useUi } from "@/store/ui";

import { activeRule, gaussian, INPUT_NAMES, paceColor, type Rule, ruleSentence } from "./fuzzy";

const X_MIN = -3;
const X_MAX = 3;
const W = 240;
const H = 56;
const sx = (x: number) => ((Math.max(X_MIN, Math.min(X_MAX, x)) - X_MIN) / (X_MAX - X_MIN)) * W;
const sy = (m: number) => H - 4 - m * (H - 10);

function curve(c: number, w: number): string {
  let d = "";
  for (let i = 0; i <= 60; i++) {
    const x = X_MIN + ((X_MAX - X_MIN) * i) / 60;
    d += `${i ? "L" : "M"}${sx(x).toFixed(1)} ${sy(gaussian(x, c, w)).toFixed(1)}`;
  }
  return d;
}

/**
 * One input's fuzzy sets: every rule's Gaussian (opacity follows its firing), the tercile cuts that name the
 * LOW / MED / HIGH terms, and a needle at the current value that glides as the replay moves.
 */
export function MembershipPlot({
  input, rules, cuts, value, phi, active, theme,
}: {
  input: number; rules: Rule[]; cuts: number[]; value: number; phi: readonly number[] | null; active: number | null; theme: Theme;
}) { // fmt: skip
  const terms = input === 0 ? ["against", "neutral", "with us"] : ["low", "med", "high"];
  const [c1 = -0.43, c2 = 0.43] = cuts;
  return (
    <div>
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-fg-muted">{INPUT_NAMES[input]}</span>
        <span className="tabular font-mono text-fg-subtle">{value >= 0 ? "+" : "−"}{Math.abs(value).toFixed(2)}σ</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H + 12}`} className="mt-1 w-full overflow-visible" aria-hidden>
        <line x1={0} x2={W} y1={H - 4} y2={H - 4} stroke="var(--hairline-strong)" />
        {[c1, c2].map((c) => (
          <line key={c} x1={sx(c)} x2={sx(c)} y1={4} y2={H - 4} stroke="var(--hairline-strong)" strokeDasharray="2 3" />
        ))}
        {rules.map((r, i) => {
          const f = phi?.[i] ?? 0;
          const on = i === active;
          return (
            <path key={r.rule} d={curve(r.centres[input] ?? 0, r.widths[input] ?? 1)} fill="none"
              stroke={paceColor(r.pace, theme)} strokeWidth={on ? 2.25 : 1.25}
              strokeOpacity={phi ? 0.18 + 0.82 * f : 0.6}
              style={{ transition: "stroke-opacity 250ms ease, stroke-width 250ms ease" }} />
          );
        })}
        {/* The needle: a transform transition makes it glide between decisions. */}
        <g style={{ transform: `translateX(${sx(value)}px)`, transition: "transform 260ms cubic-bezier(.2,.7,.2,1)" }}>
          <line x1={0} x2={0} y1={2} y2={H - 4} stroke="var(--fg)" strokeWidth={1.5} />
          <circle cx={0} cy={2} r={3} fill="var(--fg)" />
        </g>
        {[(X_MIN + c1) / 2, (c1 + c2) / 2, (c2 + X_MAX) / 2].map((x, j) => (
          <text key={terms[j]} x={sx(x)} y={H + 10} textAnchor="middle" fontSize="9" fill="var(--fg-subtle)">{terms[j]}</text>
        ))}
      </svg>
    </div>
  );
}

/** Firing strength φ of every rule as a bar coloured by the rule's pace; the active rule is outlined. */
export function RuleBars({ rules, phi, active, theme, onHover }: {
  rules: Rule[]; phi: readonly number[] | null; active: number | null; theme: Theme; onHover?: (rule: number | null) => void;
}) { // fmt: skip
  return (
    <ul className="space-y-1.5" aria-label="Rule firing strengths">
      {rules.map((r, i) => {
        const f = phi?.[i] ?? 0;
        const on = i === active;
        return (
          <li key={r.rule} onMouseEnter={() => onHover?.(i)} onMouseLeave={() => onHover?.(null)}
            className={clsx("grid grid-cols-[2.25rem_1fr_2.75rem] items-center gap-2 rounded-md px-1.5 py-1 text-xs transition-colors", on && "bg-surface-2")}>
            <span className={clsx("font-mono", on ? "text-fg" : "text-fg-subtle")}>R{r.rule}</span>
            <span className="relative h-2 overflow-hidden rounded-full bg-surface-3">
              <span className="absolute inset-y-0 left-0 rounded-full"
                style={{ width: `${(f * 100).toFixed(1)}%`, background: paceColor(r.pace, theme), transition: "width 260ms cubic-bezier(.2,.7,.2,1)" }} />
            </span>
            <span className={clsx("tabular text-right font-mono", on ? "text-fg" : "text-fg-subtle")}>{Math.round(f * 100)}%</span>
          </li>
        );
      })}
    </ul>
  );
}

/** The rule in charge, as a sentence. Keyed on the rule, so a regime change re-enters with a glow. */
export function ActiveRule({ rule, firing }: { rule: Rule; firing: number }) {
  const s = ruleSentence(rule);
  return (
    <div key={rule.rule} className="gradient-border glow-in animate-enter rounded-[var(--radius-panel)] bg-surface-2 p-3.5" data-active="true">
      <p className="flex items-center justify-between text-xs text-fg-subtle">
        <span>Active rule R{rule.rule}</span>
        <span className="tabular font-mono">fires {Math.round(firing * 100)}%</span>
      </p>
      <p className="mt-2 text-sm leading-relaxed">
        <span className="font-mono text-xs text-accent-2">IF </span>
        {s.clauses.map((c, j) => (
          <span key={c.input}>
            {j > 0 && <span className="font-mono text-xs text-fg-subtle"> ∧ </span>}
            <span className="text-fg-muted">{c.input.toLowerCase()} </span>
            <span className="font-medium text-fg">{c.term}</span>
          </span>
        ))}
        <br />
        <span className="font-mono text-xs text-accent-2">THEN </span>
        <span className="font-medium text-aurora">{s.then}</span>
      </p>
    </div>
  );
}

/** The whole fuzzy brain for one decision: active rule, firing bars and the three membership plots. */
export function FuzzyBrain({ rules, phi, regime, onHover }: {
  rules: Rules; phi: readonly number[] | null; regime: readonly number[]; onHover?: (rule: number | null) => void;
}) { // fmt: skip
  const theme = useUi((s) => s.effectiveTheme());
  const active = activeRule(phi);
  const rule = active === null ? undefined : rules.rules[active];
  return (
    <div className="space-y-4">
      {rule && <ActiveRule rule={rule} firing={phi?.[active ?? 0] ?? 0} />}
      <RuleBars rules={rules.rules} phi={phi} active={active} theme={theme} onHover={onHover} />
      <div className="space-y-3">
        {[0, 1, 2].map((j) => (
          <MembershipPlot key={j} input={j} rules={rules.rules} cuts={rules.terciles[j] ?? []} value={regime[j] ?? 0}
            phi={phi} active={active} theme={theme} />
        ))}
      </div>
    </div>
  );
}
