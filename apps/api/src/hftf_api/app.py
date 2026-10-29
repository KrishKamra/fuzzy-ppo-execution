"""FastAPI routes over `DemoService` (docs/DEMO_PLAN.md §3.2). Local only: the data is licensed.

uv run hftf serve          # http://127.0.0.1:8000, OpenAPI at /docs
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from hftf_api.schemas import Meta, OrderRequest, OrderTicket, OrderTrace, Results, Rules, RunCurves
from hftf_api.service import DemoService

DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]  # the Vite dev server (web/)


def _call(fn, *args):
    try:
        return fn(*args)
    except KeyError as err:
        raise HTTPException(status_code=404, detail=str(err.args[0])) from err
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


def create_app(service: DemoService) -> FastAPI:
    app = FastAPI(
        title="hftf demo API",
        version="0.1.0",
        description="Regime-aware execution with fuzzy PPO: order traces, rules, training and results.",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=DEV_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"]
    )

    app.state.service = service

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/meta", response_model=Meta)
    def meta():
        return service.meta()

    @app.get("/api/order/random", response_model=OrderTicket)
    def random_order(
        split: str = "val", sim: Literal["bar", "tick"] = "bar", seed: int = Query(0, ge=0, lt=2**31)
    ):
        return _call(service.random_ticket, split, sim, seed)

    @app.post("/api/order", response_model=OrderTrace)
    def order(req: OrderRequest):
        return _call(service.order, req.model_dump())

    @app.get("/api/policies/{name}/rules", response_model=Rules)
    def rules(name: str):
        return _call(service.rules, name)

    @app.get("/api/runs/{name}", response_model=RunCurves)
    def run(name: str):
        return _call(service.run, name)

    @app.get("/api/results", response_model=Results)
    def results():
        return service.result_tables()

    return app
