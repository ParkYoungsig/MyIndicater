from __future__ import annotations

import pandas as pd

from core.allocators import normalize_weights
from core.strategies.base import Strategy


class AllWeatherStrategy(Strategy):
    name = "all_weather"

    def __init__(self, target_weights: dict[str, float] | None = None) -> None:
        self.target_weights = target_weights or {}

    def generate_weights(self, prices: pd.DataFrame) -> pd.Series:
        assets = list(prices.columns)
        if self.target_weights:
            w = pd.Series({a: self.target_weights.get(a, 0.0) for a in assets})
        else:
            w = (
                pd.Series(1.0 / len(assets), index=assets)
                if assets
                else pd.Series(dtype=float)
            )
        return normalize_weights(w)
