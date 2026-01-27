from __future__ import annotations

import pandas as pd

from core.allocators import build_weights
from core.signals import average_momentum_score
from core.strategies.base import Strategy


class AMSStrategy(Strategy):
    name = "ams"

    def __init__(
        self,
        target_assets: list[str] | None = None,
        cash_asset: str | None = None,
        months: int = 12,
        trading_days_per_month: int = 21,
        normalize: bool = False,
    ) -> None:
        self.target_assets = target_assets or []
        self.cash_asset = cash_asset
        self.months = months
        self.trading_days_per_month = trading_days_per_month
        self.normalize = normalize

    def generate_weights(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty:
            return pd.Series(dtype=float)

        universe = list(prices.columns)
        assets = self.target_assets or universe
        scores = average_momentum_score(
            prices[assets],
            months=self.months,
            trading_days_per_month=self.trading_days_per_month,
        ).iloc[-1]

        if self.normalize and scores.sum() > 0:
            weights = scores / scores.sum()
        else:
            weights = scores

        targets = weights.to_dict()

        if self.cash_asset and self.cash_asset in universe:
            remaining = 1.0 - sum(targets.values())
            if remaining > 0:
                targets[self.cash_asset] = targets.get(self.cash_asset, 0.0) + remaining

        return build_weights(targets, universe)
