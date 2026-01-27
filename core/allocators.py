from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def normalize_weights(weights: pd.Series) -> pd.Series:
    w = weights.copy()
    total = w.sum()
    if total == 0:
        return w
    return w / total


def equal_weight(assets: list[str]) -> pd.Series:
    if not assets:
        return pd.Series(dtype=float)
    w = pd.Series(1.0 / len(assets), index=assets)
    return w


def inverse_vol_weights(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    vol = returns.rolling(window).std().iloc[-1]
    vol = vol.replace(0, np.nan)
    inv = 1 / vol
    return normalize_weights(inv.fillna(0.0))


def risk_parity_weights(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    return inverse_vol_weights(returns, window=window)


def gmvp_weights(
    returns: pd.DataFrame,
    window: int = 60,
    ridge: float = 1e-6,
    long_only: bool = True,
) -> pd.Series:
    data = returns.dropna()
    if data.empty:
        return pd.Series(dtype=float)
    if window is not None and window > 0:
        data = data.iloc[-window:]

    cov = data.cov()
    n = cov.shape[0]
    if n == 0:
        return pd.Series(dtype=float)

    cov = cov.values + np.eye(n) * ridge
    inv = np.linalg.pinv(cov)
    ones = np.ones(n)
    denom = ones.T @ inv @ ones
    if denom == 0:
        return equal_weight(list(data.columns))

    w = (inv @ ones) / denom
    weights = pd.Series(w, index=data.columns)
    if long_only:
        weights = weights.clip(lower=0.0)
    if weights.sum() == 0:
        return equal_weight(list(data.columns))
    return normalize_weights(weights)


def cap_weights(weights: pd.Series, cap: float = 0.3) -> pd.Series:
    w = weights.clip(upper=cap)
    return normalize_weights(w)


def apply_signal(weights: pd.Series, signal: pd.Series) -> pd.Series:
    aligned = weights.mul(signal, fill_value=0.0)
    return normalize_weights(aligned)


def rank_series(values: pd.Series, ascending: bool = False) -> pd.Series:
    return values.rank(ascending=ascending, method="first")


def select_top_n(values: pd.Series, top_n: int, ascending: bool = False) -> list[str]:
    ranked = values.sort_values(ascending=ascending)
    return list(ranked.head(top_n).index)


def single_asset_weight(asset: str, universe: list[str]) -> pd.Series:
    weights = pd.Series(0.0, index=universe)
    if asset in weights.index:
        weights.loc[asset] = 1.0
    return weights


def build_weights(targets: dict[str, float], universe: list[str]) -> pd.Series:
    weights = pd.Series(0.0, index=universe)
    for asset, weight in targets.items():
        if asset in weights.index:
            weights.loc[asset] = weight
    return normalize_weights(weights)


@dataclass(frozen=True)
class EqualWeightAllocator:
    def targets(
        self, *, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> Dict[str, float]:
        if not tickers:
            return {}
        w = 1.0 / float(len(tickers))
        return {t: w for t in tickers}


@dataclass(frozen=True)
class SlotWeightAllocator:
    slot_weights: Sequence[float]

    def targets(
        self, *, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> Dict[str, float]:
        if not tickers:
            return {}
        total = float(sum(self.slot_weights)) if self.slot_weights else 1.0
        if total == 0:
            weights = [0.0 for _ in self.slot_weights]
        else:
            weights = [float(w) / total for w in self.slot_weights]
        return {t: w for t, w in zip(tickers, weights)}


@dataclass(frozen=True)
class InverseVolatilityAllocator:
    window: int = 60
    min_vol: float = 1e-8
    max_weight: float | None = None

    def targets(
        self, *, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> Dict[str, float]:
        if not tickers:
            return {}

        window = int(self.window)
        windowed = prices[tickers].loc[:as_of].tail(window + 1)
        returns = windowed.pct_change(fill_method=None).dropna()
        if returns.empty:
            w = 1.0 / float(len(tickers))
            return {t: w for t in tickers}

        vol = (
            returns.std().replace(0.0, float(self.min_vol)).fillna(float(self.min_vol))
        )
        vol = vol.clip(lower=float(self.min_vol))
        inv = 1.0 / vol
        inv = inv.fillna(float(self.min_vol))

        weights = (
            inv / inv.sum()
            if float(inv.sum()) > 0
            else pd.Series(1.0 / len(tickers), index=tickers)
        )

        if self.max_weight is not None:
            cap = float(self.max_weight)
            weights = weights.clip(upper=cap)
            s = float(weights.sum())
            if s > 0:
                weights = weights / s

        return {t: float(weights.get(t, 0.0)) for t in tickers}
