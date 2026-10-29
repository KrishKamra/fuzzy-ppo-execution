import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router";

import { AppShell } from "@/components/shell/AppShell";
import { Overview } from "@/pages/Overview";
import { Brain, Evidence, Execute, NotFound, Training } from "@/pages/Placeholders";

export const routes = [
  {
    element: <AppShell />,
    children: [
      { path: "/", element: <Overview /> },
      { path: "/execute", element: <Execute /> },
      { path: "/brain", element: <Brain /> },
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
