# web: demo frontend

The course demo's UI (design spec: [`docs/UI_DESIGN.md`](../docs/UI_DESIGN.md); API: [`docs/DEMO_PLAN.md`](../docs/DEMO_PLAN.md)).
It runs **locally only**: the backend serves licensed Bloomberg-derived data.

```bash
# terminal 1, repo root: the API on 127.0.0.1:8000 (needs data/ and trained runs/)
uv run hftf serve

# terminal 2, web/: the dev server on http://127.0.0.1:5173 (proxies /api to the API)
npm install
npm run dev
```

| Command | What it does |
|---|---|
| `npm run dev` | dev server with hot reload |
| `npm run build` | type check + production build into `dist/` |
| `npm run preview` | serve the production build (also proxies `/api`) |
| `npm test` | Vitest + Testing Library (no backend needed; `fetch` is mocked) |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run gen:api` | regenerate `src/api/schema.d.ts` from the backend's OpenAPI schema after an API change |

Layout: `src/styles/index.css` (design tokens, themes), `src/lib/palette.ts` (the validated data palette;
do not re-order), `src/api/` (typed client and query hooks), `src/store/ui.ts` (theme, presentation mode,
palette), `src/components/shell/` (top bar, command palette, app shell), `src/pages/` (screens).

Keys: `1`-`5` screens, `Ctrl/⌘ K` command palette, `T` theme, `P` presentation mode (projector).
