from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RebalanceContext:
    date: pd.Timestamp
    prices: pd.DataFrame
    last_rebalance_date: pd.Timestamp
    last_target_weights: pd.Series
    current_weights: pd.Series


class RebalanceTrigger:
    def prepare(self, prices: pd.DataFrame) -> None:
        return None

    def should_rebalance(self, context: RebalanceContext) -> bool:
        raise NotImplementedError


class TimeTrigger(RebalanceTrigger):
    def __init__(self, freq: str = "M") -> None:
        self.freq = freq
        self._rebalance_dates: set[pd.Timestamp] = set()

    def prepare(self, prices: pd.DataFrame) -> None:
        if prices.empty:
            self._rebalance_dates = set()
            return
        dates = prices.resample(self.freq).last().index
        self._rebalance_dates = set(dates)

    def should_rebalance(self, context: RebalanceContext) -> bool:
        return context.date in self._rebalance_dates


class ThresholdTrigger(RebalanceTrigger):
    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def should_rebalance(self, context: RebalanceContext) -> bool:
        if context.last_target_weights.empty:
            return True
        drift = (context.current_weights - context.last_target_weights).abs().max()
        return float(drift) >= self.threshold
