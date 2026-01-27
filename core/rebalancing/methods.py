from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def _normalize(weights: pd.Series) -> pd.Series:
    total = weights.sum()
    if total == 0:
        return weights
    return weights / total


@dataclass(frozen=True)
class MethodContext:
    prices: pd.DataFrame
    scores: pd.Series
    cash_asset: str


class RebalanceMethod:
    def compute_target_weights(self, context: MethodContext) -> pd.Series:
        raise NotImplementedError


class AbsoluteScoreMethod(RebalanceMethod):
    def compute_target_weights(self, context: MethodContext) -> pd.Series:
        scores = context.scores.clip(lower=0.0)
        weights = _normalize(scores)
        if weights.sum() == 0:
            weights = pd.Series(0.0, index=context.scores.index)
        return _with_cash(weights, context.cash_asset)


class RankBasedMethod(RebalanceMethod):
    def __init__(self, top_n: int = 5) -> None:
        self.top_n = top_n

    def compute_target_weights(self, context: MethodContext) -> pd.Series:
        scores = context.scores.sort_values(ascending=False)
        picks = list(scores.head(self.top_n).index)
        weights = pd.Series(0.0, index=context.scores.index)
        if picks:
            w = 1.0 / len(picks)
            weights.loc[picks] = w
        return _with_cash(weights, context.cash_asset)


class RiskParityMethod(RebalanceMethod):
    def __init__(self, window: int = 60) -> None:
        self.window = window

    def compute_target_weights(self, context: MethodContext) -> pd.Series:
        returns = context.prices.pct_change().dropna()
        if returns.empty:
            weights = pd.Series(0.0, index=context.scores.index)
            return _with_cash(weights, context.cash_asset)

        vol = returns.rolling(self.window).std().iloc[-1].replace(0, pd.NA)
        adj = context.scores.clip(lower=0.0) / vol
        weights = _normalize(adj.fillna(0.0))
        return _with_cash(weights, context.cash_asset)


def _with_cash(weights: pd.Series, cash_asset: str) -> pd.Series:
    weights = weights.copy()
    if cash_asset not in weights.index:
        weights.loc[cash_asset] = 0.0
    remaining = 1.0 - weights.sum()
    if remaining > 0:
        weights.loc[cash_asset] += remaining
    return weights
