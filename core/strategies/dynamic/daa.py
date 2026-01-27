from __future__ import annotations

import pandas as pd

from core.allocators import build_weights, select_top_n
from core.signals import weighted_momentum
from core.strategies.base import Strategy


class DAAStrategy(Strategy):
    name = "daa"

    def __init__(
        self,
        aggressive_assets: list[str] | None = None,
        defensive_assets: list[str] | None = None,
        canary_assets: list[str] | None = None,
        top_n: int = 2,
        windows: tuple[int, ...] = (1, 3, 6, 12),
        weights: tuple[float, ...] = (12, 4, 2, 1),
        trading_days_per_month: int = 21,
    ) -> None:
        self.aggressive_assets = aggressive_assets or []
        self.defensive_assets = defensive_assets or []
        self.canary_assets = canary_assets or []
        self.top_n = top_n
        self.windows = windows
        self.weights = weights
        self.trading_days_per_month = trading_days_per_month

    def generate_weights(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty:
            return pd.Series(dtype=float)

        universe = list(prices.columns)
        score = weighted_momentum(
            prices,
            windows=self.windows,
            weights=self.weights,
            trading_days_per_month=self.trading_days_per_month,
        ).iloc[-1]

        aggressive = self.aggressive_assets or universe
        defensive = self.defensive_assets or universe
        canary = self.canary_assets or []

        aggressive_scores = score.reindex(aggressive).dropna()
        defensive_scores = score.reindex(defensive).dropna()
        canary_scores = score.reindex(canary).dropna()

        if aggressive_scores.empty or defensive_scores.empty:
            return pd.Series(dtype=float)

        live_canary = (canary_scores > 0).sum() if not canary_scores.empty else 2

        if live_canary >= 2:
            picks = select_top_n(aggressive_scores, self.top_n, ascending=False)
            targets = {a: 1.0 / len(picks) for a in picks}
        elif live_canary == 1:
            agg = select_top_n(aggressive_scores, 1, ascending=False)[0]
            defn = select_top_n(defensive_scores, 1, ascending=False)[0]
            targets = {agg: 0.5, defn: 0.5}
        else:
            defn = select_top_n(defensive_scores, 1, ascending=False)[0]
            targets = {defn: 1.0}

        return build_weights(targets, universe)
