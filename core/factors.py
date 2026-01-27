from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


def momentum_score(prices: pd.DataFrame, window: int = 126) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    return prices.pct_change(window).iloc[-1]


def volatility_score(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.rolling(window).std().iloc[-1]


def value_proxy_score(prices: pd.DataFrame) -> pd.Series:
    """값(밸류) 팩터 대용 스코어(스켈레톤).

    실제 밸류 지표(PER/PBR 등)가 없으므로 가격 역수 기반의 간단 proxy를 사용합니다.
    별도 밸류 데이터가 주입되면 이 함수를 대체하세요.
    """
    if prices.empty:
        return pd.Series(dtype=float)
    last = prices.iloc[-1]
    inv = 1.0 / last.replace(0, pd.NA)
    return inv.fillna(0.0)


@dataclass(frozen=True)
class SimpleMomentumRanker:
    window: int = 60

    def rank(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> pd.Series:
        window = int(self.window)
        windowed = prices[tickers].loc[:as_of].tail(window + 1)
        if windowed.empty:
            return pd.Series(dtype=float)
        mom = windowed.pct_change(window, fill_method=None).iloc[-1]
        return mom.sort_values(ascending=False)


@dataclass(frozen=True)
class VolAdjustedMomentumRanker:
    window: int = 60
    vol_window: int | None = None
    min_vol: float = 1e-8

    def rank(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> pd.Series:
        w = int(self.window)
        vw = int(self.vol_window) if self.vol_window is not None else w
        windowed = prices[tickers].loc[:as_of].tail(max(w, vw) + 1)
        if windowed.empty:
            return pd.Series(dtype=float)

        rets = windowed.pct_change(fill_method=None)
        mom = windowed.pct_change(w, fill_method=None).iloc[-1]
        vol = rets.tail(vw).std().clip(lower=float(self.min_vol))

        score = (mom / vol).fillna(0.0)
        return score.sort_values(ascending=False)
