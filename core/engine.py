from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd

from core.rebalancing.methods import (
    AbsoluteScoreMethod,
    MethodContext,
    RankBasedMethod,
    RebalanceMethod,
    RiskParityMethod,
)
from core.rebalancing.triggers import (
    RebalanceContext,
    RebalanceTrigger,
    ThresholdTrigger,
    TimeTrigger,
)
from core.scoring import FactorConfig, FactorScorer
from core.types import (
    Allocator,
    BacktestResult,
    CorrelationFilter,
    MomentumRanker,
    Selector,
    StopLoss,
)


def _legacy_to_plugin_cfg(
    logic: Dict[str, Any], dynamic: Dict[str, Any]
) -> Dict[str, Any]:
    if any(
        k in logic
        for k in [
            "correlation_filter",
            "momentum_ranker",
            "selector",
            "allocator",
            "stop_loss",
        ]
    ):
        return logic

    corr_window = int(dynamic.get("CORRELATION_WINDOW", 60))
    mom_window = int(dynamic.get("MOMENTUM_WINDOW", 60))

    return {
        "correlation_filter": {
            "type": "average",
            "threshold": float(logic.get("CORRELATION_THRESHOLD", 0.0)),
            "window": int(corr_window),
        },
        "momentum_ranker": {
            "type": "simple",
            "window": int(mom_window),
        },
        "selector": {
            "type": "rank_range",
            "rank_range": list(logic.get("MOMENTUM_RANK_RANGE", [1, 1])),
            "replacement_order": list(logic.get("REPLACEMENT_ORDER", [])),
        },
        "allocator": {
            "type": "slot_weights",
            "slot_weights": list(logic.get("SLOT_WEIGHTS", [])),
        },
        "stop_loss": {
            "type": (
                "trailing_pct" if float(logic.get("STOP_LOSS_PCT", 0.0)) > 0 else "none"
            ),
            "pct": float(logic.get("STOP_LOSS_PCT", 0.0)),
        },
    }


def build_dynamic_components(
    config: Dict[str, Any], *, fee: float, slot_count: int
) -> Tuple[
    CorrelationFilter,
    MomentumRanker,
    Selector,
    Allocator,
    StopLoss,
]:
    from core.allocators import (
        EqualWeightAllocator,
        InverseVolatilityAllocator,
        SlotWeightAllocator,
    )
    from core.factors import SimpleMomentumRanker, VolAdjustedMomentumRanker
    from core.scoring import RankRangeWithReplacementSelector, TopNSelector
    from core.signals import (
        AverageCorrelationFilter,
        NoCorrelationFilter,
        NoStopLoss,
        TrailingStopLoss,
    )

    dynamic = config["DYNAMIC"]
    legacy_logic = dynamic.get("LOGIC", {}) or {}
    logic = _legacy_to_plugin_cfg(legacy_logic, dynamic)

    corr_cfg = logic.get("correlation_filter", {}) or {}
    corr_type = str(corr_cfg.get("type", "average")).lower()
    if corr_type in ["none", "off", "disabled"]:
        correlation: CorrelationFilter = NoCorrelationFilter()
    else:
        correlation = AverageCorrelationFilter(
            threshold=float(corr_cfg.get("threshold", 0.0)),
            window=int(corr_cfg.get("window", dynamic.get("CORRELATION_WINDOW", 60))),
        )

    mom_cfg = logic.get("momentum_ranker", {}) or {}
    mom_type = str(mom_cfg.get("type", "simple")).lower()
    if mom_type in ["vol_adj", "voladjusted", "risk_adj", "sharpe_like"]:
        momentum = VolAdjustedMomentumRanker(
            window=int(mom_cfg.get("window", dynamic.get("MOMENTUM_WINDOW", 60))),
            vol_window=mom_cfg.get("vol_window"),
            min_vol=float(mom_cfg.get("min_vol", 1e-8)),
        )
    else:
        momentum = SimpleMomentumRanker(
            window=int(mom_cfg.get("window", dynamic.get("MOMENTUM_WINDOW", 60)))
        )

    sel_cfg = logic.get("selector", {}) or {}
    sel_type = str(sel_cfg.get("type", "rank_range")).lower()
    if sel_type in ["topn", "top_n", "top"]:
        selector: Selector = TopNSelector(n=int(sel_cfg.get("n", slot_count)))
    else:
        rr = sel_cfg.get("rank_range", [1, slot_count])
        start = int(rr[0]) if len(rr) >= 1 else 1
        end = int(rr[1]) if len(rr) >= 2 else start
        selector = RankRangeWithReplacementSelector(
            rank_range=(start, end),
            replacement_order=list(sel_cfg.get("replacement_order", [])),
        )

    alloc_cfg = logic.get("allocator", {}) or {}
    alloc_type = str(alloc_cfg.get("type", "slot_weights")).lower()
    if alloc_type in ["equal", "equal_weight", "equalweight"]:
        allocator: Allocator = EqualWeightAllocator()
    elif alloc_type in [
        "inv_vol",
        "inverse_vol",
        "inverse_volatility",
        "risk_parity_1overvol",
    ]:
        allocator = InverseVolatilityAllocator(
            window=int(alloc_cfg.get("window", dynamic.get("MOMENTUM_WINDOW", 60))),
            min_vol=float(alloc_cfg.get("min_vol", 1e-8)),
            max_weight=alloc_cfg.get("max_weight"),
        )
    else:
        allocator = SlotWeightAllocator(
            slot_weights=list(
                alloc_cfg.get("slot_weights", legacy_logic.get("SLOT_WEIGHTS", []))
            )
        )

    sl_cfg = logic.get("stop_loss", {}) or {}
    sl_type = str(sl_cfg.get("type", "trailing_pct")).lower()
    selection_cfg = (
        (dynamic.get("SELECTION", {}) or {}) if isinstance(dynamic, dict) else {}
    )
    reserve_enabled = bool(selection_cfg.get("reserve_enabled", False))
    replace_flag = sl_cfg.get("replace")
    if replace_flag is None:
        replace_flag = not reserve_enabled
    replace_flag = bool(replace_flag)
    if sl_type in ["none", "off", "disabled"] or float(sl_cfg.get("pct", 0.0)) <= 0:
        stop_loss: StopLoss = NoStopLoss()
    else:
        stop_loss = TrailingStopLoss(
            pct=float(sl_cfg.get("pct", 0.0)),
            fee=float(fee),
            selector=selector,
            slot_count=int(slot_count),
            replace=replace_flag,
        )

    return correlation, momentum, selector, allocator, stop_loss


@dataclass(frozen=True)
class BacktestConfig:
    scorer: FactorScorer
    cash_asset: str
    trigger: RebalanceTrigger
    method: RebalanceMethod


def build_scorer(cfg: dict[str, Any]) -> FactorScorer:
    factors_cfg = cfg.get("factors", [])
    factors: list[FactorConfig] = []
    for f in factors_cfg:
        factors.append(
            FactorConfig(
                name=str(f.get("name")),
                weight=float(f.get("weight", 1.0)),
                params=dict(f.get("params", {}) or {}),
            )
        )
    normalize = cfg.get("normalize", "zscore")
    return FactorScorer(factors, normalize=normalize)


def build_trigger(cfg: dict[str, Any]) -> RebalanceTrigger:
    t = (cfg.get("type") or "time").lower()
    if t == "threshold":
        return ThresholdTrigger(threshold=float(cfg.get("threshold", 0.05)))
    return TimeTrigger(freq=str(cfg.get("freq", "M")))


def build_method(cfg: dict[str, Any]) -> RebalanceMethod:
    m = (cfg.get("type") or "absolute_score").lower()
    if m == "rank":
        return RankBasedMethod(top_n=int(cfg.get("top_n", 5)))
    if m == "risk_parity":
        return RiskParityMethod(window=int(cfg.get("window", 60)))
    return AbsoluteScoreMethod()


def _ensure_cash_prices(prices: pd.DataFrame, cash_asset: str) -> pd.DataFrame:
    if cash_asset in prices.columns:
        return prices
    cash = pd.Series(1.0, index=prices.index, name=cash_asset)
    return prices.join(cash)


def run_backtest(prices: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    if prices.empty:
        raise ValueError("prices is empty")

    prices = prices.sort_index()
    prices = _ensure_cash_prices(prices, cfg.cash_asset)

    cfg.trigger.prepare(prices)
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    scorer = cfg.scorer

    last_rebalance_date = prices.index[0]
    last_target = pd.Series(0.0, index=prices.columns)

    for dt in prices.index:
        if dt == prices.index[0]:
            score = scorer.score(prices.loc[:dt]).scores
            method_ctx = MethodContext(
                prices=prices.loc[:dt], scores=score, cash_asset=cfg.cash_asset
            )
            last_target = (
                cfg.method.compute_target_weights(method_ctx)
                .reindex(prices.columns)
                .fillna(0.0)
            )
            weights.loc[dt] = last_target
            last_rebalance_date = dt
            continue

        price_base = prices.loc[last_rebalance_date]
        price_now = prices.loc[dt]
        rel = price_now / price_base
        current_values = last_target * rel
        current_weights = (
            current_values / current_values.sum()
            if current_values.sum() != 0
            else last_target
        )

        score = scorer.score(prices.loc[:dt]).scores
        context = RebalanceContext(
            date=dt,
            prices=prices.loc[:dt],
            last_rebalance_date=last_rebalance_date,
            last_target_weights=last_target,
            current_weights=current_weights,
        )

        if cfg.trigger.should_rebalance(context):
            method_ctx = MethodContext(
                prices=prices.loc[:dt], scores=score, cash_asset=cfg.cash_asset
            )
            last_target = (
                cfg.method.compute_target_weights(method_ctx)
                .reindex(prices.columns)
                .fillna(0.0)
            )
            last_rebalance_date = dt

        weights.loc[dt] = last_target

    returns = prices.pct_change().fillna(0.0)
    port_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    equity = (1 + port_returns).cumprod()

    return BacktestResult(equity=equity, returns=port_returns, weights=weights)
