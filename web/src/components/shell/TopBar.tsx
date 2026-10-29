import { clsx } from "clsx";
import { Command, Moon, Presentation, Sun } from "lucide-react";
import { NavLink } from "react-router";

import { useHealth } from "@/api/queries";
import { NAV } from "@/app/nav";
import { Button, Kbd } from "@/components/ui/primitives";
import { useUi } from "@/store/ui";

function ApiStatus() {
  const { isSuccess, isPending } = useHealth();
  const label = isPending ? "Connecting" : isSuccess ? "API live" : "API offline";
  return (
    <span className="hidden items-center gap-2 text-xs text-fg-muted md:inline-flex" role="status" title={label}>
      <span
        aria-hidden
        className={clsx(
          "size-2 rounded-full",
          isPending ? "bg-fg-subtle" : isSuccess ? "bg-good shadow-[0_0_8px_var(--good)]" : "bg-bad",
        )}
      />
      {label}
    </span>
  );
}

/** The mark (public/custom.svg): three fuzzy membership functions and the agent's state firing the middle one. */
export function LogoMark({ className = "size-7" }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" aria-hidden className={clsx(className, "shrink-0")}>
      <defs>
        <linearGradient id="mark-aurora" x1="4" y1="4" x2="60" y2="60" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#6366F1" />
          <stop offset="0.5" stopColor="#8B5CF6" />
          <stop offset="1" stopColor="#22D3EE" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#mark-aurora)" />
      <g fill="none" stroke="#fff" strokeLinecap="round">
        <path d="M8 46 C14 46 15 27 20 27 S26 46 33 46" strokeOpacity="0.5" strokeWidth="2.6" />
        <path d="M31 46 C38 46 39 27 44 27 S50 46 56 46" strokeOpacity="0.5" strokeWidth="2.6" />
        <path d="M16 46 C24 46 25 16 32 16 S40 46 48 46" strokeWidth="3.6" />
      </g>
      <circle cx="38.6" cy="27.4" r="6.4" fill="#fff" fillOpacity="0.22" />
      <circle cx="38.6" cy="27.4" r="3.6" fill="#fff" />
    </svg>
  );
}

export function Logo() {
  return (
    <NavLink to="/" className="group flex items-center gap-2.5" title="Overview">
      <LogoMark className="size-7 transition-transform duration-300 group-hover:rotate-[-6deg] group-hover:scale-105" />
      <span className="text-sm font-semibold tracking-tight">
        Fuzzy<span className="text-fg-subtle"> · </span>Execution
      </span>
    </NavLink>
  );
}

export function TopBar() {
  const { theme, presentation, toggleTheme, togglePresentation, setPaletteOpen } = useUi();
  return (
    <header className="glass sticky top-0 z-40 border-b border-hairline">
      <div className="mx-auto flex h-14 max-w-[1440px] items-center gap-6 px-4 sm:px-6">
        <Logo />
        <nav aria-label="Screens" className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
          {NAV.map((item, i) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === "/"}
              title={`${item.hint} (${i + 1})`}
              className={({ isActive }) =>
                clsx(
                  "relative flex h-9 items-center gap-2 rounded-[var(--radius-control)] px-3 text-sm whitespace-nowrap",
                  "transition-colors duration-150",
                  isActive ? "bg-surface-2 text-fg" : "text-fg-muted hover:text-fg",
                )
              }
            >
              {({ isActive }) => (
                <>
                  <item.icon aria-hidden className="size-4" />
                  {item.label}
                  {isActive && (
                    <span aria-hidden className="bg-aurora absolute inset-x-3 bottom-0 h-0.5 rounded-full" />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>
        <ApiStatus />
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            className="px-2.5"
            onClick={togglePresentation}
            aria-pressed={presentation}
            aria-label="Presentation mode (P)"
            title="Presentation mode (P)"
          >
            <Presentation className={clsx("size-4", presentation && "text-accent-2")} />
          </Button>
          <Button
            variant="ghost"
            className="px-2.5"
            onClick={toggleTheme}
            disabled={presentation}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme (T)`}
            title="Theme (T)"
          >
            {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>
          <Button variant="secondary" className="ml-1 gap-2" onClick={() => setPaletteOpen(true)}>
            <Command aria-hidden className="size-3.5" />
            <span className="hidden text-fg-muted sm:inline">Search</span>
            <Kbd>Ctrl K</Kbd>
          </Button>
        </div>
      </div>
    </header>
  );
}
