"""Batched execution environment (BarSim, ARCHITECTURE §4) and a single-episode gymnasium wrapper.

Step k of an episode decides at minute row0 + k + 1 from the observed row row0 + k, and the child fills
in bar row0 + k + 1 at the simulated price (impact.py). The last step force-executes the residual, so
every episode completes. Rewards are in bps of arrival notional `Q · m0`:

    r_k = −side·(p_k − m0)/m0·(q_k/Q)·1e4 − fees_k − λ·(σ_k·1e4)²·(X_{k+1}/Q)²

with `X_{k+1}` the inventory left after the child, so an episode's return is −(IS + fees + λ·risk).
With `reward.price_drift = expected` the inventory × mid-move part of the mark-to-market split is dropped (its
martingale expectation is 0), so the return is the expected cost plus the risk penalty.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from hftf_sim.config import EnvConfig, FeeConfig
from hftf_sim.episodes import Episodes, EpisodeSampler
from hftf_sim.fees import FeeModel
from hftf_sim.impact import ImpactModel
from hftf_sim.limit import bar_passive_fill
from hftf_sim.market import REGIME_COLUMNS, MarketData

# log_participation = log(Q / expected horizon volume) and minutes_left / max H tell the policy the order size
# and horizon, which set the impact vs per-child-fee trade-off (ARCHITECTURE §4.2).
PRIVATE_COLUMNS = (
    "inventory_frac", "time_frac", "is_bps", "last_action", "schedule_dev",
    "log_participation", "minutes_left",
)  # fmt: skip
NEXT_SHARE_MIN = 5
EPS = 1e-12


def obs_names(market: MarketData) -> tuple[str, ...]:
    """Observation layout: regime x (eFNN input) first, then private, then market."""
    n = len(REGIME_COLUMNS)
    return (*market.obs_columns[:n], *PRIVATE_COLUMNS, *market.obs_columns[n:], "next_5m_share")


REGIME_SLICE = slice(0, len(REGIME_COLUMNS))
SUMMARY_FIELDS = (
    "is_bps", "fees_bps", "risk_bps", "cost_bps", "vwap_slip_bps", "n_children", "executed", "passive_frac",
)  # fmt: skip


class BatchExecutionEnv:
    """N parent orders stepped together. `autoreset=True` (training) re-samples finished envs from the
    sampler; `autoreset=False` (evaluation) runs a fixed batch from `reset(episodes)` until all are done."""

    def __init__(
        self,
        market: MarketData,
        cfg: EnvConfig,
        fees: FeeModel | None = None,
        sampler: EpisodeSampler | None = None,
        n_envs: int = 64,
        autoreset: bool = True,
        randomise_impact: bool = True,
        seed: int = 0,
    ) -> None:
        self.market = market
        self.cfg = cfg
        self.fees = FeeModel(FeeConfig.load()) if fees is None else fees
        self.sampler = sampler
        self.n = n_envs
        self.autoreset = autoreset
        self.randomise_impact = randomise_impact
        self.rng = np.random.default_rng(seed)
        self.a_max = cfg.action.a_max
        self.cap = cfg.action.participation_cap
        self.risk_aversion = cfg.reward.risk_aversion
        self.reward_mode = cfg.reward.mode
        if cfg.reward.price_drift == "expected" and self.reward_mode == "arrival":
            raise ValueError("price_drift: expected needs reward mode mark_to_market or vs_vwap")
        self.drift = float(cfg.reward.price_drift == "realised")
        self.impact = ImpactModel(cfg.impact, n_envs)
        self.names = obs_names(market)
        self.signed = np.concatenate(
            [market.signed_mask[: REGIME_SLICE.stop], np.zeros(len(PRIVATE_COLUMNS), bool),
             market.signed_mask[REGIME_SLICE.stop :], [False]]
        )  # fmt: skip
        self.max_horizon = max(cfg.episode.horizons_min)
        self.ep: Episodes | None = None

    @property
    def obs_dim(self) -> int:
        return len(self.names)

    # ------------------------------------------------------------------ reset
    def reset(self, episodes: Episodes | None = None) -> np.ndarray:
        if episodes is None:
            if self.sampler is None:
                raise ValueError("reset() needs episodes or a sampler")
            episodes = self.sampler.sample(self.n, self.rng)
        if len(episodes) != self.n:
            raise ValueError(f"expected {self.n} episodes, got {len(episodes)}")
        self.ep = episodes = episodes.take(np.arange(self.n))  # own copy: autoreset writes into it
        self.k = np.zeros(self.n, dtype=np.int64)
        self.remaining = episodes.qty.astype(np.float64).copy()
        self.disp = np.zeros(self.n)
        self.shadow_left = self.remaining.copy()  # shadow VWAP (reward mode vs_vwap)
        self.shadow_disp = np.zeros(self.n)
        self.last_action = np.ones(self.n)  # a = 1 is the adaptive-VWAP prior
        self.m0 = self._arrival_mid(episodes.session, episodes.row0)
        self.m_prev = self.m0.copy()  # historical mid at the previous fill (mark-to-market reward)
        self.done = np.zeros(self.n, dtype=bool)
        self.totals = {
            f: np.zeros(self.n) for f in ("is_bps", "fees_bps", "risk_bps", "n_children", "cash", "passive")
        }
        self.mkt_value = np.zeros(self.n)
        self.mkt_volume = np.zeros(self.n)
        self.impact.draw(np.arange(self.n), episodes.seed, self.randomise_impact)
        return self.observe()

    def _reset_idx(self, idx: np.ndarray) -> None:
        new = self.sampler.sample(len(idx), self.rng)
        for f in new.__dataclass_fields__:
            getattr(self.ep, f)[idx] = getattr(new, f)
        self.k[idx] = 0
        self.remaining[idx] = new.qty
        self.disp[idx] = 0.0
        self.shadow_left[idx] = new.qty
        self.shadow_disp[idx] = 0.0
        self.last_action[idx] = 1.0
        self.m0[idx] = self._arrival_mid(new.session, new.row0)
        self.m_prev[idx] = self.m0[idx]
        self.done[idx] = False
        for v in (*self.totals.values(), self.mkt_value, self.mkt_volume):
            v[idx] = 0.0
        self.impact.draw(idx, new.seed, self.randomise_impact)

    # ------------------------------------------------------------ price clock
    def _arrival_mid(self, session: np.ndarray, row0: np.ndarray) -> np.ndarray:
        """Arrival price `m0`: the mid observed at t0 (TickSim reads it from the 1-second grid)."""
        return self.market.mid[session, row0]

    def _mark_mid(self, session: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Mid of fill bar `j` that the next step's inventory P&L is measured from (mark-to-market reward)."""
        return self.market.mid[session, j]

    # --------------------------------------------------------------- schedule
    def _fill_row(self) -> np.ndarray:
        return self.ep.row0 + np.minimum(self.k, self.ep.horizon - 1) + 1

    def _profile(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expected volume of the next fill bar, of the remaining fill bars, and of the whole horizon."""
        s, cp, j = self.ep.session, self.market.cum_prof, self._fill_row()
        end = self.ep.row0 + self.ep.horizon + 1
        w = self.market.prof[s, j]
        rest = cp[s, end] - cp[s, j]
        total = cp[s, end] - cp[s, self.ep.row0 + 1]
        return w, np.maximum(rest, EPS), np.maximum(total, EPS)

    def pace_to_shares(self, action: np.ndarray, remaining: np.ndarray | None = None) -> np.ndarray:
        """`q = clip(a · X · w_k / Σ_{s≥k} w_s, 0, min(X, cap · V̂_k))` (ARCHITECTURE §4.3)."""
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.n), 0.0, self.a_max)
        x = self.remaining if remaining is None else remaining
        w, rest, _ = self._profile()
        return np.minimum(a * x * w / rest, np.minimum(x, self.cap * w))

    def implied_pace(self, q: np.ndarray) -> np.ndarray:
        w, rest, _ = self._profile()
        vwap_q = self.remaining * w / rest
        return np.clip(np.divide(q, vwap_q, out=np.zeros(self.n), where=vwap_q > EPS), 0.0, self.a_max)

    # ------------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Pace-multiplier step (the agent's action space). `action` is the pace, shape (n,), or the pace and
        the passive fraction of the child, shape (n, 2) (limit orders, ARCHITECTURE §4.9)."""
        act = np.asarray(action, dtype=np.float64)
        passive = None
        if act.ndim == 2:
            act, passive = act[:, 0], np.clip(act[:, 1], 0.0, 1.0)
        a = np.clip(act.reshape(self.n), 0.0, self.a_max)
        return self._execute(self.pace_to_shares(a), a, passive)

    def step_shares(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Child sizes in shares (baselines); not capped by participation."""
        q = np.clip(np.asarray(q, dtype=np.float64).reshape(self.n), 0.0, None)
        return self._execute(q, self.implied_pace(q))

    def _children(self, q: np.ndarray, remaining: np.ndarray, live: np.ndarray) -> np.ndarray:
        """Whole-share children; the last step force-executes the residual."""
        forced = self.k == self.ep.horizon - 1
        q = np.where(forced, remaining, np.minimum(np.rint(q), remaining))
        return np.where(live, q, 0.0)

    def _fill(self, q: np.ndarray, remaining: np.ndarray, disp: np.ndarray, live: np.ndarray):
        """Fill price and step costs (bps of arrival notional) of children `q` taken from `remaining`, with
        `disp` the already-decayed transient displacement. Returns (price, transient kick, costs)."""
        ep, mk = self.ep, self.market
        s, j = ep.session, self._fill_row()
        m, sigma, volume = mk.mid[s, j], mk.sigma[s, j], mk.volume[s, j]
        size = self.impact.scaled_size(q, volume, mk.prof[s, j])
        temp = self.impact.temporary(sigma, size)
        price = m * (1.0 + ep.side * (disp + mk.spread[s, j] / 2 + temp))
        notional0 = ep.qty * self.m0
        left = remaining - q
        costs = {
            "is_bps": ep.side * (price - self.m0) * q / notional0 * 1e4,
            # Mark-to-market split of the same IS: child cost vs the current mid + inventory x mid change.
            # Summed over an episode the two are equal (telescoping); this one assigns credit per step.
            # price_drift = expected drops the inventory term (self.drift = 0): its martingale mean is 0.
            "mtm_bps": ep.side
            * (q * (price - m) + self.drift * remaining * (m - self.m_prev))
            / notional0
            * 1e4,
            "fees_bps": self.fees(q * price, ep.side, q > 0) / notional0 * 1e4,
            "risk_bps": self.risk_aversion * (sigma * 1e4) ** 2 * (left / ep.qty) ** 2,
        }
        costs = {k: np.where(live, v, 0.0) for k, v in costs.items()}
        kick = np.where(live, self.impact.transient_kick(sigma, size), 0.0)
        return price, kick, costs

    def _passive_fill(self, post: np.ndarray, live: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BarSim: shares filled out of `post` resting at the decision's near touch through fill bar j, and
        that (historical) limit price (limit.bar_passive_fill)."""
        ep, mk = self.ep, self.market
        s, j = ep.session, self._fill_row()
        price = np.where(ep.side > 0, mk.bid[s, j - 1], mk.ask[s, j - 1])  # the quote observed at decision
        lim = self.cfg.limit
        filled = bar_passive_fill(
            post, ep.side, price, mk.low[s, j], mk.high[s, j], mk.volume[s, j],
            lim.touch_fill_frac, lim.through_fill_frac,
        )  # fmt: skip
        return np.where(live, filled, 0.0), price

    def _add_passive(self, c: dict, filled: np.ndarray, limit_px: np.ndarray, remaining: np.ndarray) -> tuple:
        """Costs of the passive fills added to the aggressive child's `c`; `remaining` is the inventory left
        after both. The fill pays the limit price shifted by our transient displacement (the whole book is),
        and no spread or temporary impact. Returns (costs, the fill price)."""
        ep = self.ep
        price = limit_px * (1.0 + ep.side * self.disp)
        notional0 = ep.qty * self.m0
        mark = self._mark_mid(ep.session, self._fill_row())
        sigma = self.market.sigma[ep.session, self._fill_row()]
        c = dict(c)
        c["is_bps"] = c["is_bps"] + ep.side * (price - self.m0) * filled / notional0 * 1e4
        c["mtm_bps"] = c["mtm_bps"] + ep.side * filled * (price - mark) / notional0 * 1e4
        c["fees_bps"] = c["fees_bps"] + self.fees(filled * price, ep.side, filled > 0) / notional0 * 1e4
        c["risk_bps"] = np.where(
            ~self.done, self.risk_aversion * (sigma * 1e4) ** 2 * (remaining / ep.qty) ** 2, 0.0
        )
        return c, price

    def _execute(self, q: np.ndarray, pace: np.ndarray, passive: np.ndarray | None = None):
        ep, mk, live = self.ep, self.market, ~self.done
        s, j = ep.session, self._fill_row()
        decay = self.impact.decay()
        q = self._children(q, self.remaining, live)
        post = np.zeros(self.n)
        if passive is not None:  # never on the forced last step: the residual must complete
            forced = self.k == ep.horizon - 1
            dz = self.cfg.limit.dead_zone  # snap near-0 / near-1 fractions: one order, one fee
            passive = np.where(passive <= dz, 0.0, np.where(passive >= 1.0 - dz, 1.0, passive))
            post = np.where(live & ~forced, np.floor(passive * q), 0.0)
        q_aggr = q - post
        self.disp = np.where(self.k > 0, self.disp * decay, 0.0)
        price, kick, c = self._fill(q_aggr, self.remaining, self.disp, live)
        filled, cash = np.zeros(self.n), q_aggr * price
        if post.any():
            filled, limit_px = self._passive_fill(post, live)
            c, passive_px = self._add_passive(c, filled, limit_px, self.remaining - q_aggr - filled)
            cash = cash + filled * passive_px
        q = q_aggr + filled  # executed; the unfilled passive shares are cancelled and stay in inventory
        price = np.divide(cash, q, out=price.copy(), where=q > 0)
        is_bps, fees_bps, risk_bps = c["is_bps"], c["fees_bps"], c["risk_bps"]
        cost = (c["mtm_bps"] if self.reward_mode != "arrival" else is_bps) + fees_bps + risk_bps
        if self.reward_mode == "vs_vwap":
            # Benchmark-relative reward: minus the same step of a shadow adaptive VWAP (a = 1) on the same
            # path with its own impact. Its cost does not depend on our actions, so the optimal policy is
            # unchanged, but the price-drift noise now scales with our deviation from VWAP.
            qb = self._children(
                self.pace_to_shares(np.ones(self.n), self.shadow_left), self.shadow_left, live
            )
            self.shadow_disp = np.where(self.k > 0, self.shadow_disp * decay, 0.0)
            _, kick_b, cb = self._fill(qb, self.shadow_left, self.shadow_disp, live)
            cost = cost - (cb["mtm_bps"] + cb["fees_bps"] + cb["risk_bps"])
            self.shadow_left = self.shadow_left - qb
            self.shadow_disp = self.shadow_disp + kick_b
        reward = -cost
        self.remaining = self.remaining - q
        self.m_prev = np.where(live, self._mark_mid(s, j), self.m_prev)
        self.disp = self.disp + kick
        volume = mk.volume[s, j]

        t = self.totals
        t["is_bps"] += is_bps
        t["fees_bps"] += fees_bps
        t["risk_bps"] += risk_bps
        t["n_children"] += q > 0
        t["cash"] += cash
        t["passive"] += filled
        self.mkt_value += np.where(live, mk.value[s, j], 0.0)
        self.mkt_volume += np.where(live, volume, 0.0)
        self.last_action = np.where(live, pace, self.last_action)
        self.k = self.k + live
        finished = live & (self.k == ep.horizon)
        self.done |= finished

        info = {"q": q, "price": price, "is_bps": is_bps, "fees_bps": fees_bps, "risk_bps": risk_bps}
        info.update(posted=post, passive_filled=filled)
        done = finished.copy()
        if finished.any():
            idx = np.flatnonzero(finished)
            info["final_idx"] = idx
            info["final"] = self.summary(idx)
            if self.autoreset:
                self._reset_idx(idx)
        return self.observe(), reward, done, info

    # ------------------------------------------------------------ observation
    def observe(self) -> np.ndarray:
        """Observation at each env's current decision; uses rows ≤ row0 + k only (plus the causal profile)."""
        ep, mk = self.ep, self.market
        row = ep.row0 + np.minimum(self.k, ep.horizon - 1)
        feat = mk.obs[ep.session, row]
        w, rest, total = self._profile()
        cp, s, j = mk.cum_prof, ep.session, self._fill_row()
        end = ep.row0 + ep.horizon + 1
        nxt = cp[s, np.minimum(j + NEXT_SHARE_MIN, end)] - cp[s, j]
        sched = (cp[s, j] - cp[s, ep.row0 + 1]) / total
        private = np.stack(
            [
                self.remaining / ep.qty,
                (ep.horizon - self.k) / ep.horizon,
                self.totals["is_bps"],
                self.last_action,
                (ep.qty - self.remaining) / ep.qty - sched,
                np.log(ep.qty / total),
                (ep.horizon - self.k) / self.max_horizon,
            ],
            axis=1,
        )
        n = REGIME_SLICE.stop
        obs = np.concatenate([feat[:, :n], private, feat[:, n:], (nxt / rest)[:, None]], axis=1)
        return np.where(self.signed, obs * ep.side[:, None], obs)

    def summary(self, idx: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Per-episode totals (bps of arrival notional). `vwap_slip_bps` is against the market VWAP of the
        fill bars; NaN if they had no volume."""
        idx = np.arange(self.n) if idx is None else idx
        t, ep = self.totals, self.ep
        executed = ep.qty[idx] - self.remaining[idx]
        avg_px = np.divide(t["cash"][idx], executed, out=np.full(len(idx), np.nan), where=executed > 0)
        vol = self.mkt_volume[idx]
        mkt_vwap = np.divide(self.mkt_value[idx], vol, out=np.full(len(idx), np.nan), where=vol > 0)
        out = {f: t[f][idx].copy() for f in ("is_bps", "fees_bps", "risk_bps", "n_children")}
        out["cost_bps"] = out["is_bps"] + out["fees_bps"] + out["risk_bps"]
        out["vwap_slip_bps"] = ep.side[idx] * (avg_px - mkt_vwap) / mkt_vwap * 1e4
        out["executed"] = executed
        out["passive_frac"] = t["passive"][idx] / ep.qty[idx]  # share of the order filled passively
        return out


class ExecutionEnv(gym.Env):
    """Single-episode gymnasium view of the batched env (for compatibility checks, not for training)."""

    metadata = {"render_modes": []}

    def __init__(self, market: MarketData, cfg: EnvConfig, sampler: EpisodeSampler, **kwargs) -> None:
        self.batch = BatchExecutionEnv(market, cfg, sampler=sampler, n_envs=1, autoreset=False, **kwargs)
        self.sampler = sampler
        dim = self.batch.obs_dim
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float64)
        self.action_space = gym.spaces.Box(0.0, cfg.action.a_max, shape=(1,), dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        episodes = (options or {}).get("episodes")
        if episodes is None:
            episodes = self.sampler.sample(1, self.np_random)
        return self.batch.reset(episodes)[0], {}

    def step(self, action):
        obs, reward, done, info = self.batch.step(np.asarray(action).reshape(1))
        final = info.get("final")
        out = {k: v[0] for k, v in final.items()} if final is not None else {}
        return obs[0], float(reward[0]), bool(done[0]), False, out
