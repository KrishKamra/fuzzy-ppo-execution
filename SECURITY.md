# Security policy

## Supported versions

Only the `main` branch is maintained. This is a research and teaching project: it is not production trading
software and must not be connected to a live trading account.

## Reporting a vulnerability or a data leak

Please **do not open a public issue**. Report privately through GitHub's
[private vulnerability reporting](https://github.com/KrishKamra/fuzzy-ppo-execution/security/advisories/new)
(Security tab → *Report a vulnerability*). Include what you found, how to reproduce it, and its impact.
Expect an acknowledgement within a week.

Please report privately, in particular:

- **Licensed data in the repository or its history.** Prices, ticks, bars, the feature table, trained
  checkpoints, or screenshots and figures that show market data. The Bloomberg data is licensed and must not be
  redistributed; leaks are removed from history.
- **Secrets.** Any credential, token or key committed by mistake.
- **The demo API reachable beyond localhost.** `hftf serve` binds to `127.0.0.1` by design.

## Design notes

- The API (`apps/api`) is meant for local demos. It binds to `127.0.0.1`, allows CORS only for the local dev
  server, serves only configured run names (no path from a request reaches the file system), and returns
  per-order views, never bulk data.
- Checkpoints are loaded with `torch.load(weights_only=False)`, which runs pickled code. Load only checkpoints
  you trained yourself or received from a trusted teammate; never load one downloaded from an untrusted source.
- Dependencies are pinned (`uv.lock`, `web/package-lock.json`) and updated monthly by Dependabot.
