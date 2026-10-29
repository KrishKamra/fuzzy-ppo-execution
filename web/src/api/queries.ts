import { useQuery } from "@tanstack/react-query";

import { api } from "./client";

/** API liveness for the status dot: polls gently, never retries noisily. */
export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15_000, retry: false });
}

/** Sessions, policies and limits: static for a server's lifetime. */
export function useMeta() {
  return useQuery({ queryKey: ["meta"], queryFn: api.meta, staleTime: Infinity, retry: 1 });
}

/** A fuzzy policy's rules (centres, widths, terms, pace): fixed for a checkpoint. */
export function useRules(policy: string | null | undefined, enabled = true) {
  return useQuery({
    queryKey: ["rules", policy],
    queryFn: () => api.rules(policy as string),
    enabled: !!policy && enabled,
    staleTime: Infinity,
    retry: false,
  });
}
