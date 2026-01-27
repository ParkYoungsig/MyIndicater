from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _find_parquet_path(base: Path, symbol: str) -> Path | None:
    candidates = []
    if base.name in {"processed", "raw"}:
        candidates.append(base / f"{symbol}.parquet")
        candidates.append(base.parent / "processed" / f"{symbol}.parquet")
        candidates.append(base.parent / "raw" / f"{symbol}.parquet")
    else:
        candidates.extend(
            [
                base / "processed" / f"{symbol}.parquet",
                base / "raw" / f"{symbol}.parquet",
            ]
        )
    for p in candidates:
        if p.exists():
            return p

    exact = next(base.rglob(f"{symbol}.parquet"), None)
    if exact is not None:
        return exact

    matches = list(base.rglob(f"{symbol}*.parquet"))
    if not matches:
        matches = list(base.rglob(f"*{symbol}*.parquet"))
    if matches:
        matches.sort(key=lambda p: (len(p.as_posix()), p.name))
        return matches[0]

    return None


def list_local_parquet_tickers(
    data_root: str, *, exclude_dirs: list[str] | None = None
) -> list[str]:
    base = Path(data_root).resolve()
    if not base.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    roots: list[Path] = []
    if base.name in {"processed", "raw"}:
        roots = [base]
    else:
        processed = base / "processed"
        raw = base / "raw"
        if processed.exists():
            roots.append(processed)
        if raw.exists():
            roots.append(raw)
        if not roots:
            roots = [base]

    tickers: set[str] = set()
    exclude_dirs = [d.lower() for d in (exclude_dirs or [])]
    for root in roots:
        for path in root.rglob("*.parquet"):
            if not path.is_file():
                continue
            if exclude_dirs:
                parts = [p.lower() for p in path.parts]
                if any(d in parts for d in exclude_dirs):
                    continue
            tickers.add(path.stem)

    return sorted(tickers)


def load_local_parquet_frame(
    symbol: str,
    *,
    data_root: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    base = Path(data_root).resolve()
    if not base.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    path = _find_parquet_path(base, str(symbol))
    if path is None:
        raise FileNotFoundError(f"Parquet not found for {symbol}")

    df = pd.read_parquet(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    if start:
        df = df.loc[df.index >= pd.to_datetime(start)]
    if end:
        df = df.loc[df.index <= pd.to_datetime(end)]

    return df


def _read_parquet_field(
    path: Path,
    start: str | None,
    end: str | None,
    field: str,
) -> pd.Series:
    df = pd.read_parquet(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    if start:
        df = df.loc[df.index >= pd.to_datetime(start)]
    if end:
        df = df.loc[df.index <= pd.to_datetime(end)]

    field_lower = str(field).lower()
    candidates = [field_lower, field_lower.capitalize(), field_lower.upper()]
    if field_lower == "close":
        candidates.extend(["adj close", "adj_close", "Adj Close", "Adj_Close"])

    field_col = next((c for c in candidates if c in df.columns), None)
    if field_col is None:
        raise ValueError(f"{field} column not found in {path}")

    series = df[field_col].copy()
    series.name = path.stem
    return series


def load_local_parquet_prices(
    tickers: Iterable[str],
    *,
    data_root: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ffill: bool = True,
) -> pd.DataFrame:
    base = Path(data_root).resolve()
    if not base.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    frames = []
    for t in tickers:
        symbol = str(t)
        path = _find_parquet_path(base, symbol)
        if path is None:
            raise FileNotFoundError(f"Parquet not found for {symbol}")
        frames.append(_read_parquet_field(path, start, end, "close"))

    prices = pd.concat(frames, axis=1).sort_index()
    if ffill:
        prices = prices.ffill()
    return prices.dropna(how="all")


def load_local_parquet_opens(
    tickers: Iterable[str],
    *,
    data_root: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ffill: bool = False,
) -> pd.DataFrame:
    base = Path(data_root).resolve()
    if not base.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    frames = []
    for t in tickers:
        symbol = str(t)
        path = _find_parquet_path(base, symbol)
        if path is None:
            raise FileNotFoundError(f"Parquet not found for {symbol}")
        frames.append(_read_parquet_field(path, start, end, "open"))

    prices = pd.concat(frames, axis=1).sort_index()
    if ffill:
        prices = prices.ffill()
    return prices.dropna(how="all")


def load_local_parquet_field_bulk(
    tickers: Iterable[str],
    field_candidates: list[str],
    *,
    data_root: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    여러 종목의 특정 필드(예: 시가총액, 거래대금)를 한 번에 로딩해서 DataFrame으로 반환
    """
    base = Path(data_root).resolve()
    if not base.exists():
        return pd.DataFrame()

    paths_by_symbol: dict[str, Path] = {}
    for path in base.rglob("*.parquet"):
        if not path.is_file():
            continue
        symbol = path.stem
        if symbol not in paths_by_symbol:
            paths_by_symbol[symbol] = path
        else:
            current = paths_by_symbol[symbol]
            if len(path.as_posix()) < len(current.as_posix()):
                paths_by_symbol[symbol] = path

    start_time = time.perf_counter()
    start_ts = pd.to_datetime(start) if start else None
    end_ts = pd.to_datetime(end) if end else None
    field_keys = [str(c).lower() for c in field_candidates]
    date_keys = ["date", "datetime"]

    def _read_one(symbol: str) -> pd.Series | None:
        path = paths_by_symbol.get(symbol)
        if path is None:
            return None

        try:
            parquet = pq.ParquetFile(path)
            schema_names = list(parquet.schema.names)
            if not schema_names:
                return None

            lower_cols = {c.lower(): c for c in schema_names}
            date_col = next((lower_cols[k] for k in date_keys if k in lower_cols), None)
            if date_col is None:
                return None

            target_col = None
            for cand in field_keys:
                if cand in lower_cols:
                    target_col = lower_cols[cand]
                    break
            if target_col is None:
                return None

            columns = [date_col] if target_col == date_col else [date_col, target_col]
            table = parquet.read(columns=columns)
            df = table.to_pandas()
            if date_col not in df.columns:
                return None

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)

            if start_ts is not None:
                df = df.loc[df.index >= start_ts]
            if end_ts is not None:
                df = df.loc[df.index <= end_ts]

            if target_col not in df.columns:
                return None
            s = df[target_col].copy()
            s.name = symbol
            s = s[~s.index.duplicated(keep="last")]
            return s
        except Exception:
            return None

    frames: list[pd.Series] = []
    submitted = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for t in tickers:
            futures.append(executor.submit(_read_one, str(t)))
            submitted += 1
        for future in as_completed(futures):
            series = future.result()
            if series is not None and not series.empty:
                frames.append(series)

    if not frames:
        elapsed = time.perf_counter() - start_time
        logger.info(
            "bulk field load: empty (submitted=%s, elapsed=%.2fs)",
            submitted,
            elapsed,
        )
        return pd.DataFrame()

    merged = pd.concat(frames, axis=1).sort_index()
    elapsed = time.perf_counter() - start_time
    logger.info(
        "bulk field load: series=%s, submitted=%s, rows=%s, cols=%s, elapsed=%.2fs",
        len(frames),
        submitted,
        merged.shape[0],
        merged.shape[1],
        elapsed,
    )
    return merged
