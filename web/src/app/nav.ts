import { Activity, BrainCircuit, FlaskConical, LineChart, Sparkles, type LucideIcon } from "lucide-react";

/** The five screens (docs/UI_DESIGN.md §3). One list drives the top bar, the 1-5 hotkeys and the palette. */
export interface NavItem {
  path: string;
  label: string;
  hint: string;
  icon: LucideIcon;
}

export const NAV: readonly NavItem[] = [
  { path: "/", label: "Overview", hint: "The story in one screen", icon: Sparkles },
  { path: "/execute", label: "Execute", hint: "Replay a real order, minute by minute", icon: LineChart },
  { path: "/brain", label: "Brain", hint: "The fuzzy rules and the regime space", icon: BrainCircuit },
  { path: "/evidence", label: "Evidence", hint: "What the statistics support", icon: FlaskConical },
  { path: "/training", label: "Training", hint: "How the agent and its rules learned", icon: Activity },
];
