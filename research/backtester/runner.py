from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.engine import (
    BacktestConfig,
    build_method,
    build_scorer,
    build_trigger,
    run_backtest,
)
from infra.data_manager.downloader import load_ohlcv_bulk

from .config import apply_overrides, load_dual_engine_config
from .master import MasterPortfolio
from .metrics import performance_summary


def _load_snapshot_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df


@dataclass(frozen=True)
class DualEngineResult:
    config: Dict[str, Any]
    equity: pd.Series
    returns: pd.Series
    static_equity: pd.Series
    static_returns: pd.Series
    dynamic_equity: pd.Series
    dynamic_returns: pd.Series
    cash_weight: pd.Series
    static_cash_weight: pd.Series
    dynamic_cash_weight: pd.Series
    benchmark_equity: pd.Series
    benchmark_returns: pd.Series
    performance: Dict[str, float]
    benchmark_performance: Dict[str, float]
    prices: pd.DataFrame
    open_prices: pd.DataFrame
    market_cap: pd.DataFrame
    trading_value: pd.DataFrame
    dynamic_selection_log: list[Dict[str, Any]]


def run_dual_engine_backtest(
    *,
    config_name: str = "backtester",
    config: Optional[Dict[str, Any]] = None,
    logic_name: Optional[str] = None,
    json_run: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> DualEngineResult:
    cfg = (
        config
        if config is not None
        else load_dual_engine_config(
            config_name, logic_name=logic_name, json_run=json_run
        )
    )
    cfg = apply_overrides(cfg, start=start, end=end)

    # JSON replay with full data snapshot: if snapshot files exist in the referenced run_dir, use them.

    snap_prices: pd.DataFrame | None = None

    snap_open: pd.DataFrame | None = None

    snap_mc: pd.DataFrame | None = None

    snap_tv: pd.DataFrame | None = None


    json_cfg = cfg.get("JSON", {}) or {}

    run_dir_name: str | None = None

    if isinstance(json_cfg, dict) and json_cfg.get("enabled") and json_cfg.get("run_dir"):

        run_dir_name = str(json_cfg.get("run_dir"))

    elif json_run:

        run_dir_name = str(json_run)


    if run_dir_name:

        run_dir = (Path(__file__).resolve().parents[2] / "results" / run_dir_name).resolve()

        prices_path = run_dir / "prices_close.csv"

        open_path = run_dir / "prices_open.csv"

        mc_path = run_dir / "market_cap.csv"

        tv_path = run_dir / "trading_value.csv"


        if prices_path.exists():

            snap_prices = _load_snapshot_csv(prices_path)

        if open_path.exists():

            snap_open = _load_snapshot_csv(open_path)

        if mc_path.exists():

            snap_mc = _load_snapshot_csv(mc_path)

        if tv_path.exists():

            snap_tv = _load_snapshot_csv(tv_path)


    master = MasterPortfolio(

        cfg,

        prices=snap_prices,

        open_prices=snap_open,

        market_cap=snap_mc,

        trading_value=snap_tv,

    )

    equity = master.run()
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    static_equity = master.static_equity
    dynamic_equity = master.dynamic_equity
    static_returns = static_equity.pct_change(fill_method=None).fillna(0.0)
    dynamic_returns = dynamic_equity.pct_change(fill_method=None).fillna(0.0)

    benchmark_ticker = cfg.get("BENCHMARK_TICKER")
    prices = master.prices
    if benchmark_ticker and benchmark_ticker in prices.columns:
        bench = prices[benchmark_ticker].dropna()
        benchmark_equity = (bench / bench.iloc[0]) * float(cfg["INITIAL_CAPITAL"])
        benchmark_returns = benchmark_equity.pct_change(fill_method=None).fillna(0.0)
    else:
        benchmark_equity = pd.Series(dtype=float)
        benchmark_returns = pd.Series(dtype=float)

    perf = performance_summary(equity, returns, int(cfg.get("TRADING_DAYS", 252)))
    bench_perf = performance_summary(
        benchmark_equity, benchmark_returns, int(cfg.get("TRADING_DAYS", 252))
    )

    return DualEngineResult(
        config=cfg,
        equity=equity,
        returns=returns,
        static_equity=static_equity,
        static_returns=static_returns,
        dynamic_equity=dynamic_equity,
        dynamic_returns=dynamic_returns,
        cash_weight=master.cash_weight,
        static_cash_weight=master.static_cash_weight,
        dynamic_cash_weight=master.dynamic_cash_weight,
        benchmark_equity=benchmark_equity,
        benchmark_returns=benchmark_returns,
        performance=perf,
        benchmark_performance=bench_perf,
        prices=prices,
        dynamic_selection_log=list(getattr(master.dynamic_engine, "selection_log", [])),
    )


def resolve_universe(
    backtest_cfg: dict[str, Any], universe_cfg: dict[str, Any]
) -> list[str]:
    universe_key = backtest_cfg.get("universe_key")
    if universe_key:
        universe = universe_cfg.get(universe_key, [])
    else:
        universe = backtest_cfg.get("universe", [])
    if not universe:
        raise ValueError("backtest.yaml에 universe 또는 universe_key가 비어 있습니다.")
    return list(universe)


def load_prices(
    universe: list[str],
    data_cfg: dict[str, Any],
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    data = load_ohlcv_bulk(
        universe,
        start=start,
        end=end,
        source=data_cfg.get("source", "fdr"),
        use_cache=data_cfg.get("use_cache", True),
        cache_dir=data_cfg.get("cache_dir", "data/raw"),
        preprocess=data_cfg.get("preprocess", True),
    )
    return pd.concat({k: v["close"] for k, v in data.items()}, axis=1).dropna()


def build_backtest_config(backtest_cfg: dict[str, Any]) -> BacktestConfig:
    scoring_cfg = backtest_cfg.get("scoring", {})
    scorer = build_scorer(scoring_cfg)
    cash_asset = scoring_cfg.get("cash_asset", "CASH")
    trigger = build_trigger(backtest_cfg.get("rebalancing", {}).get("trigger", {}))
    method = build_method(backtest_cfg.get("rebalancing", {}).get("method", {}))
    return BacktestConfig(
        scorer=scorer, cash_asset=cash_asset, trigger=trigger, method=method
    )


def run_cio_backtest(
    backtest_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    universe_cfg: dict[str, Any],
    start: str | None,
    end: str | None,
):
    universe = resolve_universe(backtest_cfg, universe_cfg)
    prices = load_prices(universe, data_cfg, start, end)
    cfg = build_backtest_config(backtest_cfg)
    return run_backtest(prices, cfg)
