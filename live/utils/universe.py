from __future__ import annotations

from typing import Callable, List

import pandas as pd

from infra.logger import get_logger


logger = get_logger("live.universe")


def get_target_universe(
    *,
    min_mcap: float = 1_000_000_000_000,  # 1,000억 KRW (example)
    min_volume: float = 100_000,  # 100k shares
    markets: List[str] | None = None,
    exclude_regex: str = r"스팩|우선주|리츠|ETN|ETF",
    progress: Callable[[str], None] | None = None,
) -> List[str]:
    """Return a list of ticker codes filtered by marketcap and recent volume.

    The function fetches a single FDR StockListing('KRX') snapshot (fast),
    applies numeric coercions and the requested filters, and returns codes.

    Args:
        min_mcap: minimum market capitalization (in KRW units as in FDR 'Marcap').
        min_volume: minimum recent volume (as provided by FDR 'Volume').
        markets: list of markets to include (defaults to ['KOSPI','KOSDAQ']).
        exclude_regex: regex to exclude names (SPACs, preferred, etc.).
        progress: optional callback accepting a status string for progress updates.

    Returns:
        list of ticker codes (strings) that passed the filters.
    """
    if markets is None:
        markets = ["KOSPI", "KOSDAQ"]

    def _p(msg: str) -> None:
        logger.info(msg)
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    _p("⚡ [1/4] FDR: Fetching KRX listing snapshot...")
    try:
        import FinanceDataReader as fdr

        df = fdr.StockListing("KRX")
    except Exception as exc:  # pragma: no cover - environment/runtime
        _p(f"FDR listing failed: {exc}")
        raise

    _p(f"⚡ [2/4] Snapshot fetched: total_rows={len(df)}")

    # normalize expected columns
    df = df.copy()
    # Common column names: Code, Name, Market, Marcap, Volume, Close
    for col in ("Marcap", "Volume", "Close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Market filter
    df = df[df["Market"].isin(markets)] if "Market" in df.columns else df
    _p(f"⚡ [3/4] After market filter: rows={len(df)}")

    # Apply numeric filters
    cond = pd.Series(True, index=df.index)
    if "Marcap" in df.columns:
        cond = cond & (df["Marcap"] >= float(min_mcap))
    if "Volume" in df.columns:
        cond = cond & (df["Volume"] >= float(min_volume))
    if "Name" in df.columns and exclude_regex:
        cond = cond & (~df["Name"].str.contains(exclude_regex, na=False))

    filtered = df[cond]
    _p(f"⚡ [4/4] After numeric/exclude filters: rows={len(filtered)}")

    codes_col = None
    for c in ("Code", "code", "Symbol", "symbol"):
        if c in filtered.columns:
            codes_col = c
            break

    if codes_col is None:
        _p("No code column found in FDR snapshot.")
        return []

    codes = [str(x).zfill(6) for x in filtered[codes_col].tolist()]
    _p(f"✅ Selected {len(codes)} tickers.")
    return codes
