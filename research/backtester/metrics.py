from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def performance_summary(
    equity: pd.Series, returns: pd.Series, trading_days: int = 252
) -> Dict[str, float]:
    if equity.empty:
        return {"CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "TotalReturn": 0.0}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    running_max = equity.cummax()
    mdd = (equity / running_max - 1).min()

    if returns.std() == 0 or pd.isna(returns.std()):
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / returns.std()) * (trading_days**0.5)

    return {
        "CAGR": float(cagr),
        "MDD": float(mdd),
        "Sharpe": float(sharpe),
        "TotalReturn": float(total_return),
    }


def cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def mdd(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    excess = returns - (rf / periods_per_year)
    if excess.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def summary(equity: pd.Series, returns: pd.Series) -> dict:
    return {
        "CAGR": cagr(equity),
        "MDD": mdd(equity),
        "Sharpe": sharpe(returns),
        "TotalReturn": (
            equity.iloc[-1] / equity.iloc[0] - 1 if not equity.empty else 0.0
        ),
    }
