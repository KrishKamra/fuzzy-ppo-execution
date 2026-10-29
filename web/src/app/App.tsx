import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router";

import { AppShell } from "@/components/shell/AppShell";
import { Overview } from "@/pages/Overview";
import { Evidence, NotFound, Training } from "@/pages/Placeholders";

export const routes = [
  {
    element: <AppShell />,
    children: [
      { path: "/", element: <Overview /> },
      // Charts (ECharts) load with the screens that draw them, not with the shell.
      { path: "/execute", lazy: () => import("@/pages/Execute").then((m) => ({ Component: m.Execute })) },
      { path: "/brain", lazy: () => import("@/pages/Brain").then((m) => ({ Component: m.Brain })) },
      { path: "/evidence", element: <Evidence /> },
      { path: "/training", element: <Training /> },
      { path: "*", element: <NotFound /> },
    ],
  },
];

export function makeQueryClient() {
  return new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false } } });
}

export function App() {
  return (
    <QueryClientProvider client={makeQueryClient()}>
      <RouterProvider router={createBrowserRouter(routes)} />
    </QueryClientProvider>
  );
}
