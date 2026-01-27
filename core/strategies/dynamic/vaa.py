from __future__ import annotations

import pandas as pd

from core.allocators import single_asset_weight
from core.signals import weighted_momentum
from core.strategies.base import Strategy


class VAAStrategy(Strategy):
    name = "vaa"

    def __init__(
        self,
        aggressive_assets: list[str] | None = None,
        defensive_assets: list[str] | None = None,
        windows: tuple[int, ...] = (1, 3, 6, 12),
        weights: tuple[float, ...] = (12, 4, 2, 1),
        trading_days_per_month: int = 21,
    ) -> None:
        self.aggressive_assets = aggressive_assets or []
        self.defensive_assets = defensive_assets or []
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

        aggressive_scores = score.reindex(aggressive).dropna()
        defensive_scores = score.reindex(defensive).dropna()

        if aggressive_scores.empty or defensive_scores.empty:
            return pd.Series(dtype=float)

        if (aggressive_scores <= 0).any():
            target = defensive_scores.idxmax()
        else:
            target = aggressive_scores.idxmax()

        return single_asset_weight(target, universe)
