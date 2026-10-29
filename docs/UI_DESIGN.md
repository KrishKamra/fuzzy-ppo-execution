# UI design spec (M6c-M6f)

The frontend (`web/`) of the course demo. Backend and data contract: [DEMO_PLAN.md](DEMO_PLAN.md) and
`apps/api` (OpenAPI at `/docs`). This file fixes the look, the screens and the budget before any code.

## 1. Direction: "quiet precision, one vivid accent"

The product is an *agent you can read*. The UI should feel like a precision instrument (Linear, Vercel,
Raycast, Stripe's dashboards): dark, calm, dense but never cluttered, with one vivid signature gradient that
marks the intelligent parts. Polish comes from typography, spacing, motion and restraint, not from effects
on every surface.

| Effect | Verdict | Where |
|---|---|---|
| **Gradient** | **Yes, one signature gradient** ("aurora": indigo → violet → cyan), used sparingly | hero, primary button, the active-rule highlight, focus glow, a faint ambient mesh behind the app at ~6% opacity. In data: only meaningful scales (slow → fast pace) and soft area fills |
| **Glassmorphism** | **Yes, on floating chrome only** | top bar, replay transport dock, tooltips, command palette, hero cards: surfaces with content moving *beneath* them, where blur adds depth |
| Glass on chart panels | **No** | blur under lines and labels lowers contrast (WCAG), and repainting `backdrop-filter` under an animating replay costs frames. Panels are solid elevated surfaces with 1 px hairline gradient borders |
| **3D** | **Yes, in one view, where it is information** | the *regime space*: the eFNN's input is three-dimensional, so each rule is a real ellipsoid (its 1σ contour) and the order's market state a moving point. Lazy-loaded; nowhere else |
| 3D decoration (hero objects, 3D bar charts) | **No** | costs GPU and accessibility, adds no information, and 3D bars distort comparisons |
| Neumorphism / heavy shadows | **No** | low contrast, dated |
| Motion | **Yes, purposeful** | replay animation, number tickers, layout transitions, rule firing; all off under `prefers-reduced-motion` |

## 2. Design system

- **Theme:** dark by default, plus a light theme. A **presentation mode** (key `P`) switches to the light theme
  with 125% type and hides secondary chrome: dark UIs wash out on projectors, so the classroom demo needs this.
- **Colour tokens** (CSS variables, both themes): background `#0A0B10`; surface 1/2/3 as stepped elevations;
  hairline borders at 8% white; text at 92/64/42% for primary/secondary/tertiary. Accent: the aurora gradient
  `#6366F1 → #8B5CF6 → #22D3EE`.
- **Data palette:** colour-blind-safe and validated (dataviz skill) in both themes. Our policies take the accent
  hues; **VWAP is always neutral grey** (the reference); the other baselines take muted tones. Buy/sell are blue /
  orange plus an arrow icon, never red/green alone. Pace uses a diverging scale: slow (blue) → schedule (grey) →
  fast (amber).
- **Typography:** Inter Variable for UI; JetBrains Mono for numbers in tables and tickers; tabular numerals
  everywhere a number changes. Scale 12 / 13 / 15 / 18 / 24 / 32 / 48. Self-hosted via fontsource (local-only app).
- **Shape and space:** 4 px grid; radius 10 (panels) / 8 (controls) / 999 (chips); 1 px borders; one soft
  shadow level for floating chrome only.
- **Components:** shadcn/ui on Radix primitives (accessible dialogs, tabs, tooltips, sliders), restyled to the
  tokens. Icons: Lucide.

## 3. Screens

1. **Hero (landing).** One screen: "An execution agent you can read." Animated aurora mesh, three headline
   numbers that count up (beats VWAP by 0.16-0.24 bps out of sample, reproduced independently; limit orders
   beat the naive passive rule by 0.43-0.46 bps; rules that match execution theory), and **Run a live order**.
2. **Execute (the main screen).** Everything is synchronised to one replay clock.
   - *Left:* the order ticket (stock, date, start, horizon, side, size, simulator, policies) and **Surprise me**
     (`/api/order/random`). Test sessions sit behind a labelled toggle.
   - *Centre:* the price chart (mid with a bid/ask band, the 30-minute pre-order context dimmed, fills as dots
     sized by shares, hollow = passive), and below it, on a shared time axis, *inventory left vs the VWAP
     schedule* and *cost race* (running cost − VWAP; below zero = winning).
   - *Right rail: the fuzzy brain.* Three input gauges, each drawing the rules' Gaussian membership curves with
     the live value as a moving needle; a bar per rule for its firing strength φ; and the **active rule as a
     sentence**: "IF spread variance HIGH ∧ volatility HIGH THEN pace 0.90 (slow down)". A rule switch animates.
   - *Bottom: glass transport dock*, with play/pause, 1×/4×/16× speed, a scrubber with event ticks (big child,
     rule switch, passive fill), and a live leaderboard of the selected policies' cost so far.
3. **Brain (deep dive).**
   - The **3D regime space**: axes = the three standardised inputs with LOW/MED/HIGH tercile planes; each rule a
     translucent ellipsoid coloured by its pace; the current order's path as a glowing trail with the
     firing rule lit. Orbit, hover a rule to see its sentence. 2D fallback: three pairwise projections.
   - The rule table, and the membership functions per input.
   - With limit orders: each rule's passive fraction as well as its pace.
4. **Evidence.** Forest plots (point ± CI) for every claim, with session and date CIs and the Holm p beside each.
   Toggles: **our run vs the independent replication**, **BarSim vs TickSim**. Then the robustness curves
   (transient-impact strength and persistence) and the limit-order panel (learned vs naive passive). It ends
   with the claims checklist from DEMO_PLAN §5, including what is *not* established (eFNN > MLP).
5. **Training.** Validation edge over iterations with `best.pt` marked; rule count with add/merge/prune events;
   edit KL against the 0.02 guard; PPO KL.

Global: command palette (⌘K / Ctrl+K: "random order", "go to Evidence", "presentation mode"), keyboard
(Space play, ← → step, 1-5 screens), deep links (an order ticket encoded in the URL, so a demo moment can be
bookmarked).

## 4. Stack and budget

| Concern | Choice | Why |
|---|---|---|
| App | React 19 + Vite 8 + TypeScript 5.9 (strict); React Router 7 | fast dev server, typed end to end |
| Types | `openapi-typescript` from `/openapi.json` | one contract with the backend |
| Server state | TanStack Query | caching, request states |
| Replay state | Zustand (one clock store) | every panel reads one minute index; no prop drilling |
| Styling | Tailwind CSS + CSS-variable tokens, shadcn/ui (Radix) | consistent tokens, accessible primitives |
| Charts | ECharts 5, modular imports, canvas renderer, a custom theme | linked axes, smooth with many series, small when tree-shaken |
| 3D | three.js via react-three-fiber + drei, **lazy chunk**, `frameloop="demand"` | renders only on change; zero cost until opened |
| Motion | Framer Motion (layout, tickers); rAF for the replay clock | |
| Tests | Vitest + Testing Library; Playwright smoke (one order end to end) | |

**Budget** (gzip): app shell + charts ≤ 350 KB on first load, 3D chunk ≤ 200 KB on demand. First paint
< 1 s, and an order response is drawn < 400 ms after the click (the API answers in ~0.3 s for 8 policies on
CPU). 60 fps replay on a laptop iGPU, with the 3D view paused when off-screen.

**Accessibility:** WCAG AA contrast in both themes; full keyboard paths; visible focus (the accent glow);
charts carry a text summary; reduced motion respected; colour never the only signal.

## 5. Build order

| | Deliverable | Exit criterion |
|---|---|---|
| **M6c** ✅ | `web/` scaffold, tokens and themes, shell (top bar, routing, palette), typed API client | Lighthouse ≥ 95 (performance, accessibility) on the shell |
| **M6d** ✅ | Execute screen: ticket, price/inventory/cost charts, transport dock, leaderboard | a real val order replays end to end, all panels in sync |
| **M6e** ✅ | Fuzzy brain rail + Brain screen (3D regime space with 2D fallback) | rule text equals `hftf eval rules`; the 3D chunk loads only on that screen |
| **M6f** ✅ | Evidence and Training screens, hero, presentation mode, demo script, README | a 3-minute scripted demo runs from a clean checkout plus local `data/` and `runs/` |

**M6c as built:** `web/` (see `web/README.md`). Tokens in `src/styles/index.css` (Tailwind 4 `@theme`); the data
palette in `src/lib/palette.ts`, validated with the dataviz skill on our surfaces (dark: all checks pass,
worst adjacent CVD ΔE 9.4; light: separation passes, three slots under 3:1 on white, so light-mode charts
using them carry direct labels or a table). Tertiary text alphas were raised to the computed WCAG minimum
on every surface level (dark 0.50, light 0.60). Measured on the production build with the real API:
Lighthouse desktop 100 performance / 100 accessibility / 100 best practices on Overview and Execute; LCP
0.5 s, CLS ≤ 0.004; JS 132.6 KB + CSS 8.4 KB gzip. React Router 7 and TypeScript 5.9 were pinned over the
newer majors (8, 7) for tooling compatibility.

**M6d as built:** `web/src/features/execute/`. A pure model (`model.ts`: axis mapping, progressive reveal,
standings, decisions) feeds ECharts option builders (`charts.ts`), so the logic is tested without a canvas. One
replay clock (`store/replay.ts`, 2 decisions/s at 1×, 4× and 16×) drives every panel. The ticket lives in the URL
(bookmarkable; `?random=1` draws one). VWAP is always requested as the reference. Changes from the spec, found
by looking at the real render: **no end labels** (with six policies, every line ends at the same minute and
the labels collided), replaced by one HTML legend whose click sets the focus policy. The standings table is the
table view that the light-mode relief rule requires, and shows short names with the qualifier on a second
line. The Execute route is lazy, so ECharts loads only there (shell 133.8 KB + Execute 173.2 KB gzip). ECharts
is pinned to 5.6.0 (the 6.x line changed defaults). Under `prefers-reduced-motion` the replay opens at the
finished state instead of autoplaying. Measured with a real 6-policy TickSim order: Lighthouse desktop 100 /
100 / 100, LCP 0.7 s, TBT 0 ms, CLS 0.012.

**M6e as built:** `web/src/features/brain/`. `fuzzy.ts` (Gaussian membership, active rule, rule as a sentence,
pace colour) feeds `FuzzyBrain` (the active rule as a sentence; firing bars; three membership plots in plain
SVG, with the live value as a gliding needle). It appears in the Execute rail for single fuzzy checkpoints;
an MLP or an ensemble gets a note on why it has no rule base. The Brain screen draws the **3D regime space**
in plain three.js (no react-three-fiber or drei): rules as ±1σ ellipsoids coloured by pace, the tercile
planes, the order's path (line + dots, violet → cyan) and its moving point. It renders on demand, stops
off-screen, auto-rotates until touched (not under reduced motion), and hovering a rule links the ellipsoid,
bars and table. Without WebGL it falls back to two SVG projections. The pace colour scale spans ±10% (learned
paces lie in 0.85-1.1; ±15% left most rules grey).
Motion, all in CSS (opacity/transform only, off under reduced motion): screen enter, ambient drift, primary
sheen, press feedback, gradient border on hover, active-rule glow, skeleton shimmer. **Dropped as not worth
the weight:** Framer Motion, row-reorder (FLIP) animation, count-up numbers, tilt/parallax/particles.
Bundles (gzip): shell 131 KB; Execute 171 KB (ECharts); Brain 3 KB + RegimeSpace 143 KB (three.js), each
loaded only on its screen. Lighthouse desktop: Overview 100/100/100, Execute 98/100/100, Brain 95/100/100
(TBT 150 ms under software WebGL in headless Edge). Brand: `web/public/custom.svg` (logo and favicon: three
membership functions with the agent's state) and `docs/assets/hero.svg` (the README banner).

**M6f as built:** the Evidence screen (claims checklist with status icons; SVG forest plots with session and date CIs
and Holm p; our run beside the replication; BarSim vs TickSim; limit orders; robustness sweeps; by regime) and the
Training screen (validation edge with the selected checkpoint, rule-base events, PPO and edit KL against the 0.02
guard), both lazy. Confidence bands on numeric axes are custom polygon series (`components/charts/bands.ts`):
stacking two lines only works on category axes, which the first render exposed. Every number comes from the API's
report tables (`configs/demo.yaml`); the replication's values, whose rows are not kept, come from RESULTS.md.
Lighthouse desktop: Evidence 99/100/100, Training 99/100/100. `hftf report figures` writes the same evidence as
slide figures (`docs/figures/`), and `docs/DEMO_SCRIPT.md` is the 3-minute walkthrough.

**Stretch (after M6f):** to animate the rule base *growing during training* in 3D, the trainer would also need
to save rule snapshots (centres, widths) at each structure edit. Today `events.csv` records which rules
changed but not where they were. That is a small trainer change plus a re-train.
