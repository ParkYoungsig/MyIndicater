from __future__ import annotations

from typing import Iterable

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """결측치/중복 제거 기본 정제"""
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "close",
        "adj_close": "close",
        "volume": "volume",
        "tradevolume": "volume",
    }
    df = df.rename(columns=rename_map)
    return df


def preprocess_ohlcv(
    df: pd.DataFrame, required_cols: Iterable[str] = ("open", "high", "low", "close")
) -> pd.DataFrame:
    """OHLCV 데이터 전처리"""
    df = clean_data(df)
    df = _standardize_columns(df)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            pass

    df = df.sort_index()

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.loc[~df.index.isna()]
    df = df.ffill()
    df = df.dropna(subset=list(required_cols))
    return df
