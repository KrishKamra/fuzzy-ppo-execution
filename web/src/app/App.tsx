import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router";

import { AppShell } from "@/components/shell/AppShell";
import { Overview } from "@/pages/Overview";
import { NotFound } from "@/pages/NotFound";

export const routes = [
  {
    element: <AppShell />,
    children: [
      { path: "/", element: <Overview /> },
      // Charts (ECharts) load with the screens that draw them, not with the shell.
      { path: "/execute", lazy: () => import("@/pages/Execute").then((m) => ({ Component: m.Execute })) },
      { path: "/brain", lazy: () => import("@/pages/Brain").then((m) => ({ Component: m.Brain })) },
      { path: "/evidence", lazy: () => import("@/pages/Evidence").then((m) => ({ Component: m.Evidence })) },
      { path: "/training", lazy: () => import("@/pages/Training").then((m) => ({ Component: m.Training })) },
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
