from __future__ import annotations

import pandas as pd

from core.allocators import build_weights, rank_series, select_top_n
from core.signals import correlation_sum, momentum, volatility_score
from core.strategies.base import Strategy


class FAAStrategy(Strategy):
    name = "faa"

    def __init__(
        self,
        lookback: int = 84,
        vol_window: int = 84,
        corr_window: int = 84,
        top_n: int = 3,
        cash_asset: str | None = None,
        w_mom: float = 1.0,
        w_vol: float = 0.5,
        w_corr: float = 0.5,
    ) -> None:
        self.lookback = lookback
        self.vol_window = vol_window
        self.corr_window = corr_window
        self.top_n = top_n
        self.cash_asset = cash_asset
        self.w_mom = w_mom
        self.w_vol = w_vol
        self.w_corr = w_corr

    def generate_weights(self, prices: pd.DataFrame) -> pd.Series:
        if prices.empty:
            return pd.Series(dtype=float)

        universe = list(prices.columns)
        returns = prices.pct_change().dropna()

        mom = momentum(prices, window=self.lookback).iloc[-1]
        vol = volatility_score(returns, window=self.vol_window)
        corr = correlation_sum(returns, window=self.corr_window)

        rank_m = rank_series(mom, ascending=False)
        rank_v = rank_series(vol, ascending=True)
        rank_c = rank_series(corr, ascending=True)

        score = (rank_m * self.w_mom) + (rank_v * self.w_vol) + (rank_c * self.w_corr)
        pick = select_top_n(score, self.top_n, ascending=True)

        targets = {asset: 1.0 / len(pick) for asset in pick}
        weights = build_weights(targets, universe)

        if self.cash_asset and self.cash_asset in universe:
            for asset in pick:
                if mom.get(asset, 0.0) <= 0:
                    weights.loc[asset] = 0.0
            remaining = 1.0 - weights.sum()
            if remaining > 0:
                weights.loc[self.cash_asset] += remaining
        return weights
