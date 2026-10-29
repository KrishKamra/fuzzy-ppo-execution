"""Typed request and response models: the API contract (OpenAPI) the frontend's TypeScript types come from.

Units: prices in ₹, costs in bps of arrival notional, times as IST "HH:MM", minutes since 09:15 (`minute`).
Missing values (NaN in the data) are `null`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Num = float | None


class PolicyInfo(BaseModel):
    name: str
    label: str
    kind: Literal["baseline", "ppo"]
    description: str
    explains: bool  # a single fuzzy checkpoint: its trace carries rule firing (phi)
    limit_orders: bool


class SessionInfo(BaseModel):
    security: str
    date: str
    split: str
    ticks: bool  # TickSim available (val/test tick sessions)


class Meta(BaseModel):
    securities: list[str]
    sessions: list[SessionInfo]
    horizons: list[int]
    earliest_start: str
    latest_end: str
    participation: tuple[float, float]
    policies: list[PolicyInfo]
    default_policies: list[str]
    runs: list[str]
    regime_inputs: list[str]


class OrderTicket(BaseModel):
    security: str
    date: str = Field(description="IST session date, YYYY-MM-DD")
    start: str = Field(description="first decision, IST HH:MM")
    horizon: int
    side: Literal["buy", "sell"]
    participation: float = Field(gt=0.0, le=0.3, description="Q / expected volume over the horizon")


class OrderRequest(OrderTicket):
    policies: list[str] = Field(default_factory=list, description="names from /meta; empty = defaults")
    sim: Literal["bar", "tick"] = "bar"


class Order(BaseModel):
    security: str
    date: str
    split: str
    start: str
    end: str
    horizon: int
    side: Literal["buy", "sell"]
    qty: float
    arrival_mid: float
    sim: Literal["bar", "tick"]


class MarketSeries(BaseModel):
    minute: list[int]
    time: list[str]
    mid: list[Num]
    bid: list[Num]
    ask: list[Num]
    volume: list[Num]
    expected_volume: list[Num]


class Steps(BaseModel):
    minute: list[int]
    pace: list[float]
    passive: list[float]
    q: list[float]
    posted: list[float]
    passive_filled: list[float]
    price: list[float]
    remaining: list[float]
    cum_is_bps: list[float]
    cum_fees_bps: list[float]
    cum_risk_bps: list[float]
    cum_cost_bps: list[float]
    regime: list[list[float]]
    phi: list[list[float]] | None = None


class PolicyTrace(BaseModel):
    name: str
    label: str
    summary: dict[str, Num]
    steps: Steps


class OrderTrace(BaseModel):
    order: Order
    market: MarketSeries
    regime_inputs: list[str]
    policies: list[PolicyTrace]


class Rule(BaseModel):
    rule: int
    centres: list[float]  # train-standardised units, one per regime input
    widths: list[float]
    terms: list[str]  # linguistic term per input (from train terciles)
    pace: float  # firing-weighted mean pace on val
    passive: Num = None  # firing-weighted mean passive fraction (limit-order policies)
    share: float  # mean firing on val


class Rules(BaseModel):
    policy: str
    inputs: list[str]
    input_labels: list[str]
    terciles: list[list[float]]  # per input: the two cut points between LOW/MED/HIGH
    rules: list[Rule]


class RunCurves(BaseModel):
    run: str
    progress: dict[str, list[Num]]
    events: list[dict[str, str | float | int | None]]


class ResultTable(BaseModel):
    id: str
    title: str
    columns: list[str]
    rows: list[dict[str, str | float | int | None]]


class Results(BaseModel):
    tables: list[ResultTable]
