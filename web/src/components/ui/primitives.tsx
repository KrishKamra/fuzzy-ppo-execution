import { clsx } from "clsx";
import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";

type Variant = "primary" | "secondary" | "ghost";

const VARIANTS: Record<Variant, string> = {
  // The one gradient control on a screen: the primary action.
  primary: "bg-aurora text-white shadow-[0_8px_24px_-12px_var(--accent-2)] hover:brightness-110",
  secondary: "bg-surface-2 text-fg border border-hairline hover:bg-surface-3",
  ghost: "text-fg-muted hover:text-fg hover:bg-surface-2",
};

export function Button({
  variant = "secondary",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      type="button"
      className={clsx(
        "inline-flex h-9 items-center gap-2 rounded-[var(--radius-control)] px-3.5 text-sm font-medium",
        "transition-[background,color,filter] duration-150 disabled:pointer-events-none disabled:opacity-50",
        VARIANTS[variant],
        className,
      )}
      {...props}
    />
  );
}

export function Panel({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={clsx("panel", className)} {...props} />;
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-hairline bg-surface-2 px-1.5 py-0.5 font-mono text-[0.7rem] text-fg-muted">
      {children}
    </kbd>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="text-xs font-medium tracking-[0.14em] text-fg-subtle uppercase">{children}</p>;
}

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div>
        <Eyebrow>{eyebrow}</Eyebrow>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">{title}</h1>
      </div>
      {children}
    </header>
  );
}
