from __future__ import annotations

import argparse
from io import BytesIO, StringIO
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import requests


def _resolve_path(path_value: str) -> Path | None:
    if not path_value or not str(path_value).strip():
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / path).resolve()


def _resolve_input_path(path_value: str) -> Path:
    path = _resolve_path(path_value)
    if path is None:
        return Path(path_value)
    if path.exists():
        return path
    project_root = Path(__file__).resolve().parents[1]
    alt_root = project_root.parent
    alt = (alt_root / path_value).resolve()
    if alt.exists():
        return alt
    return path


def _normalize_code(series: pd.Series) -> pd.Series:
    codes = series.astype(str).str.replace(".0", "", regex=False).str.strip()
    return codes.str.zfill(6)


def _has_sector_info(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if "Sector" in df.columns and df["Sector"].notna().any():
        return True
    if "Industry" in df.columns and df["Industry"].notna().any():
        return True
    return False


def _fetch_listing_krx_excel() -> pd.DataFrame:
    url = (
        "https://kind.krx.co.kr/corpgeneral/corpList.do"
        "?method=download&searchType=13"
    )
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        content = resp.content
        try:
            df = pd.read_excel(BytesIO(content), engine="openpyxl")
        except Exception:
            tables = pd.read_html(StringIO(resp.text))
            if not tables:
                raise ValueError("KRX HTML table not found")
            df = tables[0]
    except Exception as exc:  # noqa: BLE001
        print(f"[Warning] Failed to read KRX Excel: {exc}")
        return pd.DataFrame(columns=["Code", "Sector", "Industry"])

    rename_map = {}
    for col in df.columns:
        if col in {"종목코드", "종목 코드", "코드"}:
            rename_map[col] = "Code"
        elif col in {"업종", "업종명"}:
            rename_map[col] = "Industry"
        elif col in {"회사명", "종목명", "회사 명"}:
            rename_map[col] = "Name"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Code" not in df.columns:
        return pd.DataFrame(columns=["Code", "Sector", "Industry"])

    df["Code"] = _normalize_code(df["Code"])
    cols = [c for c in ["Code", "Industry"] if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["Code", "Sector", "Industry"])
    return df[cols]


def _fetch_listing(listing_csv: Path | None = None) -> pd.DataFrame:
    if listing_csv is not None and listing_csv.exists():
        return pd.read_csv(listing_csv)

    sources = ["KRX-DESC", "KOSPI", "KOSDAQ", "KONEX", "KRX"]
    frames: list[pd.DataFrame] = []
    last_err: Exception | None = None
    for src in sources:
        try:
            df = fdr.StockListing(src)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue

    if not frames:
        print("[Warning] Failed to fetch KRX listing from FinanceDataReader.")
        if last_err is not None:
            print(f"[Warning] Last error: {last_err}")
        return pd.DataFrame(columns=["Code", "Sector", "Industry"])

    combined = pd.concat(frames, axis=0, ignore_index=True)
    if "Code" in combined.columns:
        combined["Code"] = _normalize_code(combined["Code"])
        combined = combined.drop_duplicates(subset=["Code"], keep="first")
    if not _has_sector_info(combined):
        print("[Warning] Listing fetched but Sector/Industry info is empty.")
        excel_df = _fetch_listing_krx_excel()
        if _has_sector_info(excel_df):
            combined = combined.merge(
                excel_df, on="Code", how="left", suffixes=("", "_excel")
            )
            if "Industry_excel" in combined.columns and "Industry" in combined.columns:
                combined["Industry"] = combined["Industry"].fillna(
                    combined["Industry_excel"]
                )
                combined = combined.drop(columns=["Industry_excel"])
        else:
            print("[Warning] KRX Excel did not provide Industry data.")
    return combined


def build_sector_map(
    input_csv: Path,
    output_csv: Path | None,
    output_map: Path | None,
    output_json: Path | None,
    listing_csv: Path | None = None,
    use_krx_excel: bool = True,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input not found: {input_csv}")

    base = pd.read_csv(input_csv)
    if "Code" not in base.columns:
        raise ValueError("Input csv must contain Code column")

    base["Code"] = _normalize_code(base["Code"])

    listing = _fetch_listing(listing_csv)
    if use_krx_excel and not _has_sector_info(listing):
        excel_df = _fetch_listing_krx_excel()
        if _has_sector_info(excel_df):
            listing = excel_df

    if "Code" not in listing.columns:
        raise ValueError("Listing data missing Code column")

    listing["Code"] = _normalize_code(listing["Code"])
    sector_col = "Sector" if "Sector" in listing.columns else None
    industry_col = "Industry" if "Industry" in listing.columns else None

    cols = ["Code"]
    if sector_col:
        cols.append(sector_col)
    if industry_col:
        cols.append(industry_col)

    listing = listing[cols]
    merged = base.merge(listing, on="Code", how="left")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_csv, index=False)

    if output_map is not None:
        output_map.parent.mkdir(parents=True, exist_ok=True)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)

    sector_series = None
    if sector_col and industry_col:
        sector_series = (
            merged[sector_col]
            .astype("string")
            .fillna(merged[industry_col].astype("string"))
        )
    elif sector_col:
        sector_series = merged[sector_col]
    elif industry_col:
        sector_series = merged[industry_col]

    if sector_series is not None:
        sector_series = (
            sector_series.astype("string")
            .fillna("Unknown")
            .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        )
        name_series = (
            merged["Name"].astype("string")
            if "Name" in merged.columns
            else pd.Series(["Unknown"] * len(merged), dtype="string")
        )
        market_series = (
            merged["Market"].astype("string")
            if "Market" in merged.columns
            else pd.Series(["Unknown"] * len(merged), dtype="string")
        )
        stocks_series = (
            merged["Stocks"]
            if "Stocks" in merged.columns
            else pd.Series([None] * len(merged))
        )
        sector_map = pd.DataFrame(
            {
                "Code": merged["Code"],
                "Name": name_series,
                "Market": market_series,
                "Stocks": stocks_series,
                "Sector": sector_series,
            }
        )
        if output_map is not None:
            sector_map.to_csv(output_map, index=False)
        if output_json is not None:
            sector_map.dropna().set_index("Code")["Sector"].to_json(
                output_json, force_ascii=False, indent=2
            )

    if output_csv is not None:
        print(f"[Saved] {output_csv}")
    if sector_series is not None:
        if output_map is not None:
            print(f"[Saved] {output_map}")
        if output_json is not None:
            print(f"[Saved] {output_json}")
    else:
        print("[Warning] Sector/Industry column not found in listing data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sector map from KRX master")
    parser.add_argument(
        "--input",
        type=str,
        default="../data/raw/krx_master_latest.csv",
        help="Input KRX master CSV",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Merged output CSV (optional)",
    )
    parser.add_argument(
        "--output-map",
        type=str,
        default="../data/processed/sector_map.csv",
        help="Code->Sector CSV",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Code->Sector JSON (optional)",
    )
    parser.add_argument(
        "--listing-csv",
        type=str,
        default=None,
        help="Optional listing CSV with Code/Sector/Industry columns",
    )
    parser.add_argument(
        "--no-krx-excel",
        action="store_true",
        help="Disable KRX Excel fallback",
    )
    args = parser.parse_args()

    build_sector_map(
        input_csv=_resolve_input_path(args.input),
        output_csv=_resolve_path(args.output_csv),
        output_map=_resolve_path(args.output_map),
        output_json=_resolve_path(args.output_json),
        listing_csv=_resolve_path(args.listing_csv) if args.listing_csv else None,
        use_krx_excel=not args.no_krx_excel,
    )


if __name__ == "__main__":
    main()
