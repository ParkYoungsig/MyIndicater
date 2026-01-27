"""Generate stop-loss report for given tickers.

Usage examples:
  python scripts/stop_loss_report.py --tickers 005930,000660 --pct 0.05 --out stop_report.csv
  python scripts/stop_loss_report.py --file tickers.txt --pct 0.05

Behavior:
- Determine current season -> quarter start date (Q1=Jan1, Q2=Apr1, Q3=Jul1, Q4=Oct1) based on `as_of` (defaults to today).
- For each ticker, use FinanceDataReader to fetch historical OHLCV (3 years default) and find the first trading day on/after quarter start as entry.
- Entry price = close on entry day. Stop price = entry * (1 - pct).
- Scan forward and record first date where close <= stop price.
- Output CSV with results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import os
from typing import List, Optional

import pandas as pd

# Default embedded parameters when no CLI args provided
DEFAULT_TICKERS = [
    "207940",
    "068270",
    "326030",
    "006280",
    "145020",
    "128940",
    "069620",
    "003090",
    "302440",
    "000100",
]
DEFAULT_REF = dt.date(2025, 10, 1)
DEFAULT_END = dt.date(2025, 12, 31)


def get_quarter_start(as_of: dt.date) -> dt.date:
    m = as_of.month
    y = as_of.year
    if 1 <= m <= 3:
        return dt.date(y, 1, 1)
    if 4 <= m <= 6:
        return dt.date(y, 4, 1)
    if 7 <= m <= 9:
        return dt.date(y, 7, 1)
    return dt.date(y, 10, 1)


def load_ohlcv(
    ticker: str,
    start: str,
    end: str,
    data_dir: Optional[str] = None,
    prefer_local: bool = False,
) -> pd.DataFrame:
    """Load OHLCV either from local parquet (preferred) or FinanceDataReader.

    Local path convention: <data_dir>/<TICKER>.parquet
    Example: C:/Users/dudals/Downloads/Team_Project/data/processed/kr_stocks/005930.parquet
    """
    # try local parquet first when requested
    if prefer_local and data_dir:
        local_path = os.path.join(data_dir, f"{ticker}.parquet")
        exists = os.path.exists(local_path)
        print(f"Looking for local parquet at: {local_path} (exists={exists})")
        if exists:
            try:
                size = os.path.getsize(local_path)
            except Exception:
                size = None
            print(f"Local file size: {size}")
            try:
                df_local = pd.read_parquet(local_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read local parquet {local_path}: {e}"
                ) from e
            if df_local is None or df_local.empty:
                raise RuntimeError(
                    f"Local parquet {local_path} read but contains no data"
                )
            return df_local

    try:
        import FinanceDataReader as fdr
    except Exception as e:
        raise RuntimeError(
            "FinanceDataReader is required. Install with `pip install finance-datareader`"
        ) from e
    # fdr expects ticker code format like '005930'
    return fdr.DataReader(ticker, start, end)


def next_trading_day(df: pd.DataFrame, date: dt.date) -> Optional[pd.Timestamp]:
    idx = pd.DatetimeIndex(df.index)
    d_ts = pd.to_datetime(date)
    ge = idx[idx >= d_ts]
    if ge.empty:
        return None
    return ge[0]


def analyze_ticker(
    ticker: str,
    ref_date: dt.date,
    end_date: dt.date,
    pct: float,
    data_dir: Optional[str] = None,
    prefer_local: bool = False,
) -> dict:
    """Analyze stop-loss for ticker.

    - `ref_date`: entry reference date (use next trading day if market closed)
    - `end_date`: last date to search for stop (inclusive)
    - entry price uses the `Open` (시가) on entry day
    - compare against close prices for stop condition
    """
    start_date = (ref_date - dt.timedelta(days=365 * 4)).strftime(
        "%Y-%m-%d"
    )  # fetch 4 years for safety
    end_date_str = (end_date + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        df = load_ohlcv(
            ticker,
            start_date,
            end_date_str,
            data_dir=data_dir,
            prefer_local=prefer_local,
        )
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

    if df is None or df.empty:
        return {"ticker": ticker, "error": "no data"}

    # find first trading day on/after ref_date
    entry_idx = next_trading_day(df, ref_date)
    if entry_idx is None:
        return {
            "ticker": ticker,
            "ref_date": ref_date.strftime("%Y-%m-%d"),
            "error": "no entry trading day",
        }

    # entry price = Open
    entry_price = None
    for col in ("Open", "open"):
        if col in df.columns:
            entry_price = float(df.loc[entry_idx][col])
            break
    if entry_price is None:
        # fallback to first numeric column value
        entry_price = float(df.iloc[df.index.get_loc(entry_idx)].iloc[0])

    threshold = entry_price * (1.0 - pct)

    # search forward for first close <= threshold up to end_date
    df_after = df.loc[entry_idx : pd.to_datetime(end_date)]
    stop_idx = None
    stop_price = None
    for i, row in df_after.iterrows():
        close_val = None
        for col in ("Close", "close", "Adj Close", "AdjClose"):
            if col in df_after.columns:
                close_val = float(row[col])
                break
        if close_val is None:
            close_val = float(row.iloc[0])
        if close_val <= threshold:
            stop_idx = i
            stop_price = close_val
            break

    result = {
        "ticker": ticker,
        "ref_date": ref_date.strftime("%Y-%m-%d"),
        "entry_date": entry_idx.strftime("%Y-%m-%d"),
        "entry_price": entry_price,
        "stop_pct": pct,
        "threshold_price": threshold,
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    if stop_idx is None:
        # not stopped by end_date
        result.update(
            {
                "stop_date": end_date.strftime("%Y-%m-%d"),
                "stop_price": None,
                "days_to_stop": None,
                "status": "not_stopped",
            }
        )
    else:
        days = (pd.to_datetime(stop_idx).date() - pd.to_datetime(entry_idx).date()).days
        result.update(
            {
                "stop_date": pd.to_datetime(stop_idx).strftime("%Y-%m-%d"),
                "stop_price": stop_price,
                "days_to_stop": int(days),
                "status": "stopped",
            }
        )
    return result


def parse_ticker_list(s: Optional[str], file: Optional[str]) -> List[str]:
    tickers: List[str] = []
    if s:
        tickers += [t.strip() for t in s.split(",") if t.strip()]
    if file:
        if not os.path.exists(file):
            raise FileNotFoundError(file)
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t:
                    tickers.append(t)
    return tickers


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers", help="comma separated tickers (e.g. 005930,000660)"
    )
    parser.add_argument("--file", help="file with one ticker per line")
    parser.add_argument(
        "--pct", type=float, default=0.05, help="stop-loss percent (e.g. 0.05 for 5%%)"
    )
    parser.add_argument(
        "--ref", help="reference/entry date YYYY-MM-DD (defaults to today)"
    )
    parser.add_argument("--end", help="end date YYYY-MM-DD (defaults to today)")
    parser.add_argument(
        "--data_dir",
        help=(
            "local parquet folder for tickers. "
            "Example: C:\\Users\\dudals\\Downloads\\Team_Project\\data\\processed\\kr_stocks"
        ),
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="prefer local parquet files in --data_dir over remote FinanceDataReader",
    )
    parser.add_argument("--out", help="output CSV file", default="stop_loss_report.csv")
    args = parser.parse_args(argv)

    tickers = parse_ticker_list(args.tickers, args.file)
    if not tickers:
        tickers = DEFAULT_TICKERS.copy()
        print(
            f"No tickers provided — using defaults ({len(tickers)}): {', '.join(tickers)}"
        )

    if args.ref:
        ref_date = dt.datetime.strptime(args.ref, "%Y-%m-%d").date()
    else:
        ref_date = DEFAULT_REF

    if args.end:
        end_date = dt.datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = DEFAULT_END

    def run_stop_loss(
        tickers_list: List[str],
        pct: float,
        ref_dt: dt.date,
        end_dt: dt.date,
        out_path: str,
        data_dir: Optional[str] = None,
        prefer_local: bool = False,
    ) -> pd.DataFrame:
        rows = []
        for t in tickers_list:
            try:
                r = analyze_ticker(
                    t,
                    ref_dt,
                    end_dt,
                    float(pct),
                    data_dir=data_dir,
                    prefer_local=prefer_local,
                )
                rows.append(r)
                print(f"Processed {t}: {r.get('status', r.get('error'))}")
            except Exception as e:
                print(f"Error processing {t}: {e}")
                rows.append({"ticker": t, "error": str(e)})
        df_out = pd.DataFrame(rows)
        df_out.to_csv(out_path, index=False)
        print("Wrote", out_path)
        return df_out

    # run
    run_stop_loss(
        tickers,
        args.pct,
        ref_date,
        end_date,
        args.out,
        data_dir=args.data_dir,
        prefer_local=args.local,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
