from __future__ import annotations

import pandas as pd

from core.allocators import apply_signal, equal_weight
from core.signals import trend_signal
from core.strategies.base import Strategy


class GTAAStrategy(Strategy):
    name = "gtaa"

    def __init__(self, trend_window: int = 200) -> None:
        self.trend_window = trend_window

    def generate_weights(self, prices: pd.DataFrame) -> pd.Series:
        trend = trend_signal(prices, window=self.trend_window).iloc[-1]
        base_w = equal_weight(list(prices.columns))
        return apply_signal(base_w, trend)
