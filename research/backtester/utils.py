from __future__ import annotations

from typing import Dict

import pandas as pd


def to_date_index(prices: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(prices.index, pd.DatetimeIndex):
        return pd.to_datetime(prices.index)
    return prices.index


def freq_to_pandas(freq: str) -> str:
    mapping = {
        "Daily": "D",
        "Weekly": "W",
        # pandas 2.2+ deprecation 대응
        "Monthly": "ME",
        "Quarterly": "QE",
    }
    return mapping.get(freq, freq)


def calc_nav(prices: pd.Series, holdings: Dict[str, float], cash: float) -> float:
    total = float(cash)
    for symbol, shares in holdings.items():
        px = prices.get(symbol, 0.0)
        try:
            px_f = float(px)
        except (TypeError, ValueError):
            px_f = 0.0
        if pd.isna(px_f):
            px_f = 0.0
        total += px_f * float(shares)
    return float(total)


def apply_fee(value: float, fee: float) -> float:
    return abs(value) * fee


def month_day_in_season(date: pd.Timestamp, start_md: str, end_md: str) -> bool:
    md = date.strftime("%m-%d")
    if start_md <= end_md:
        return start_md <= md <= end_md
    return md >= start_md or md <= end_md
