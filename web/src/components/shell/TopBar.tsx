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

export function Logo() {
  return (
    <NavLink to="/" className="flex items-center gap-2.5" title="Overview">
      <span aria-hidden className="bg-aurora grid size-7 place-items-center rounded-lg">
        <svg viewBox="0 0 24 24" className="size-4 text-white" fill="none" stroke="currentColor" strokeWidth="2.2">
          <path d="M3 17c3-6 6-6 9-2s6 4 9-4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
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
