from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import pandas as pd

from core.factors import momentum_score, value_proxy_score, volatility_score
from core.types import ScoreResult


def _zscore(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    std = series.std()
    if std == 0:
        return series * 0
    return (series - series.mean()) / std


@dataclass(frozen=True)
class FactorConfig:
    name: str
    weight: float
    params: dict[str, Any]


class FactorScorer:
    def __init__(
        self, factors: list[FactorConfig], normalize: str | None = "zscore"
    ) -> None:
        self.factors = factors
        self.normalize = normalize

    def _compute_factor(
        self,
        name: str,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        params: dict[str, Any],
    ) -> pd.Series:
        if name == "momentum":
            return momentum_score(prices, window=int(params.get("window", 126)))
        if name == "volatility":
            return volatility_score(returns, window=int(params.get("window", 60)))
        if name == "value":
            return value_proxy_score(prices)
        raise ValueError(f"Unknown factor: {name}")

    def score(self, prices: pd.DataFrame) -> ScoreResult:
        returns = prices.pct_change().dropna()
        components: dict[str, pd.Series] = {}

        for factor in self.factors:
            s = self._compute_factor(factor.name, prices, returns, factor.params)
            if self.normalize == "zscore":
                s = _zscore(s)
            components[factor.name] = s * factor.weight

        if not components:
            return ScoreResult(scores=pd.Series(dtype=float), components={})

        first = next(iter(components.values()))
        score = pd.Series(0.0, index=first.index)
        for s in components.values():
            score = score.add(s, fill_value=0.0)
        return ScoreResult(scores=score, components=components)


@dataclass(frozen=True)
class TopNSelector:
    n: int

    def select(self, ranked: pd.Series, slot_count: int) -> List[str]:
        n = min(int(self.n), int(slot_count))
        return list(ranked.index[:n])


@dataclass(frozen=True)
class RankRangeWithReplacementSelector:
    rank_range: Tuple[int, int] = (1, 1)
    replacement_order: Sequence[int] = ()

    def select(self, ranked: pd.Series, slot_count: int) -> List[str]:
        start, end = self.rank_range
        ranked_list = list(ranked.index)
        selected: List[str] = []

        for r in range(int(start), int(end) + 1):
            idx = r - 1
            if 0 <= idx < len(ranked_list):
                selected.append(ranked_list[idx])

        for r in self.replacement_order:
            idx = int(r) - 1
            if 0 <= idx < len(ranked_list):
                t = ranked_list[idx]
                if t not in selected:
                    selected.append(t)
            if len(selected) >= int(slot_count):
                break

        if len(selected) < int(slot_count):
            for t in ranked_list:
                if t not in selected:
                    selected.append(t)
                if len(selected) >= int(slot_count):
                    break

        return selected[: int(slot_count)]
