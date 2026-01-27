from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from core.types import DynamicState, Selector


def calc_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return prices.pct_change(periods=periods)


def rolling_vol(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return returns.rolling(window).std()


def sma(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return prices.rolling(window).mean()


def ema(prices: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    return prices.ewm(span=span, adjust=False).mean()


def momentum(prices: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    return prices.pct_change(window)


def trend_signal(prices: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    ma = sma(prices, window)
    return (prices > ma).astype(float)


def cross_over_signal(short_ma: pd.DataFrame, long_ma: pd.DataFrame) -> pd.DataFrame:
    return (short_ma > long_ma).astype(float)


def rank_signal(values: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rank = values.rank(axis=1, ascending=False, method="first")
    return (rank <= top_n).astype(float)


def zscore(values: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    mean = values.rolling(window).mean()
    std = values.rolling(window).std()
    return (values - mean) / std.replace(0, np.nan)


def weighted_momentum(
    prices: pd.DataFrame,
    windows: tuple[int, ...] = (1, 3, 6, 12),
    weights: tuple[float, ...] = (12, 4, 2, 1),
    trading_days_per_month: int = 21,
) -> pd.DataFrame:
    if len(windows) != len(weights):
        raise ValueError("windows and weights length must match")
    score = prices.copy() * 0
    for m, w in zip(windows, weights):
        ret = prices.pct_change(m * trading_days_per_month)
        score = score + (ret * w)
    return score


def average_momentum_score(
    prices: pd.DataFrame,
    months: int = 12,
    trading_days_per_month: int = 21,
) -> pd.DataFrame:
    scores: list[pd.DataFrame] = []
    for m in range(1, months + 1):
        ret = prices.pct_change(m * trading_days_per_month)
        scores.append((ret > 0).astype(float))
    if not scores:
        return prices.copy() * 0
    total = scores[0]
    for s in scores[1:]:
        total = total + s
    return total / float(months)


def volatility_score(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    return returns.rolling(window).std().iloc[-1]


def correlation_sum(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    windowed = returns.tail(window)
    corr = windowed.corr()
    return corr.sum(axis=1)


@dataclass(frozen=True)
class NoCorrelationFilter:
    def filter(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> List[str]:
        return tickers


@dataclass(frozen=True)
class AverageCorrelationFilter:
    threshold: float = 0.0
    window: int = 60

    def filter(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> List[str]:
        subset = prices[tickers].loc[:as_of].tail(int(self.window))
        returns = subset.pct_change(fill_method=None).dropna()
        if returns.empty or len(tickers) <= 1:
            return tickers

        corr = returns.corr()
        avg_corr = corr.mean(axis=1)
        filtered = avg_corr[avg_corr >= float(self.threshold)].index.tolist()
        return filtered if filtered else tickers


@dataclass(frozen=True)
class NoStopLoss:
    def apply(
        self, *, state: DynamicState, prices: pd.Series, ranked: pd.Series
    ) -> DynamicState:
        return state


@dataclass(frozen=True)
class TrailingStopLoss:
    pct: float
    fee: float
    selector: Selector
    slot_count: int
    replace: bool = True

    def apply(
        self, *, state: DynamicState, prices: pd.Series, ranked: pd.Series
    ) -> DynamicState:
        stop_pct = float(self.pct)
        if stop_pct <= 0:
            return state

        cash = float(state.cash)
        holdings = dict(state.holdings)
        high_water = dict(state.high_water)

        for symbol in list(holdings.keys()):
            px = float(prices.get(symbol, 0.0))
            if px <= 0 or pd.isna(px):
                continue

            hwm = high_water.get(symbol, px)
            hwm = max(float(hwm), px)
            high_water[symbol] = hwm

            if px <= hwm * (1 - stop_pct):
                shares = float(holdings.pop(symbol, 0.0))
                trade_value = shares * px
                cash += trade_value - (abs(trade_value) * float(self.fee))
                high_water.pop(symbol, None)

                if self.replace:
                    candidates: List[str] = self.selector.select(
                        ranked, int(self.slot_count)
                    )
                    for cand in candidates:
                        if cand not in holdings:
                            holdings[cand] = 0.0
                            cand_px = float(prices.get(cand, 0.0))
                            high_water[cand] = (
                                cand_px
                                if (cand_px > 0 and not pd.isna(cand_px))
                                else 0.0
                            )
                            break

        return DynamicState(cash=cash, holdings=holdings, high_water=high_water)
