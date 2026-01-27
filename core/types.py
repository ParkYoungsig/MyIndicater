from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol

import pandas as pd


@dataclass(frozen=True)
class ScoreResult:
    scores: pd.Series
    components: dict[str, pd.Series]


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame


@dataclass
class DynamicState:
    cash: float
    holdings: Dict[str, float]
    high_water: Dict[str, float]


class CorrelationFilter(Protocol):
    def filter(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> List[str]: ...


class MomentumRanker(Protocol):
    def rank(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> pd.Series: ...


class Selector(Protocol):
    def select(self, ranked: pd.Series, slot_count: int) -> List[str]: ...


class Allocator(Protocol):
    def targets(
        self, *, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> Dict[str, float]: ...


class StopLoss(Protocol):
    def apply(
        self, *, state: DynamicState, prices: pd.Series, ranked: pd.Series
    ) -> DynamicState: ...
