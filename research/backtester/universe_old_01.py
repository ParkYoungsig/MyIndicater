from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_loader import (
    list_local_parquet_tickers,
    load_local_parquet_opens,
    load_local_parquet_prices,
    load_local_parquet_field_bulk,
)


class UniverseLoader:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._cached_market_cap: Optional[pd.DataFrame] = None
        self._cached_trading_value: Optional[pd.DataFrame] = None

    def _resolve_data_root(self, override_root: str | None = None) -> Path:
        data_cfg = self.config.get("DATA", {}) or {}
        data_root = override_root
        if not data_root and isinstance(data_cfg, dict):
            data_root = data_cfg.get("root") or data_cfg.get("base_dir")
        if not data_root:
            data_root = "data"

        data_root_path = Path(str(data_root))
        if not data_root_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            data_root_path = (project_root / data_root_path).resolve()
        if data_root_path.name not in {"processed", "raw"}:
            processed_dir = data_root_path / "processed"
            if processed_dir.exists():
                data_root_path = processed_dir
        return data_root_path

    def _effective_start_date(self) -> str | None:
        start = self.config.get("START_DATE")
        if not start:
            return None

        dynamic_cfg = self.config.get("DYNAMIC", {}) or {}
        sel_cfg = dynamic_cfg.get("SELECTION", {}) or {}
        mode = str(sel_cfg.get("mode", "season")).lower()
        if mode != "auto":
            return str(start)

        lookback_years = int(sel_cfg.get("lookback_years", 3))
        momentum_window = int(
            sel_cfg.get("momentum_window", dynamic_cfg.get("MOMENTUM_WINDOW", 60))
        )
        corr_window = int(
            sel_cfg.get("corr_window", dynamic_cfg.get("CORRELATION_WINDOW", 60))
        )
        extra_days = max(momentum_window, corr_window) + 5
        start_ts = pd.to_datetime(start) - pd.DateOffset(years=lookback_years)
        start_ts = start_ts - pd.Timedelta(days=int(extra_days))
        return start_ts.strftime("%Y-%m-%d")

    def _dynamic_tickers(self) -> List[str]:
        tickers: List[str] = []
        dynamic_cfg = self.config.get("DYNAMIC", {}) or {}
        seasons = dynamic_cfg.get("SEASONS", []) or []
        for season in seasons:
            tickers.extend(season.get("tickers", []) or [])
        return list(dict.fromkeys(tickers))

    def dynamic_universe_tickers(self) -> List[str]:
        dynamic_cfg = self.config.get("DYNAMIC", {}) or {}
        sel_cfg = dynamic_cfg.get("SELECTION", {}) or {}
        mode = str(sel_cfg.get("mode", "season")).lower()
        if mode == "auto":
            override_root = sel_cfg.get("universe_root") or sel_cfg.get("data_root")
            data_root = self._resolve_data_root(
                override_root=str(override_root) if override_root else None
            )
            exclude_dirs: list[str] = []
            if bool(sel_cfg.get("exclude_kr_etf", True)):
                exclude_dirs.append("kr_etf")
            return list_local_parquet_tickers(str(data_root), exclude_dirs=exclude_dirs)

        return self._dynamic_tickers()

    def _static_tickers(self) -> List[str]:
        return list(self.config["STATIC"]["ASSETS"].values())

    def all_tickers(self) -> List[str]:
        tickers = self._static_tickers() + self.dynamic_universe_tickers()
        if self.config.get("BENCHMARK_TICKER"):
            tickers.append(self.config["BENCHMARK_TICKER"])
        return list(dict.fromkeys(tickers))

    def load_prices(self) -> pd.DataFrame:
        data_root_path = self._resolve_data_root()
        start_date = self._effective_start_date()

        prices = load_local_parquet_prices(
            self.all_tickers(),
            data_root=str(data_root_path),
            start=start_date,
            end=self.config.get("END_DATE"),
            ffill=bool(self.config.get("FFILL", True)),
        )
        return prices.dropna(how="all")

    def load_open_prices(self) -> pd.DataFrame:
        data_root_path = self._resolve_data_root()
        start_date = self._effective_start_date()

        prices = load_local_parquet_opens(
            self.all_tickers(),
            data_root=str(data_root_path),
            start=start_date,
            end=self.config.get("END_DATE"),
            ffill=False,
        )
        return prices.dropna(how="all")

    def correlation_filter(
        self,
        prices: pd.DataFrame,
        tickers: List[str],
        as_of: pd.Timestamp,
    ) -> List[str]:
        logic = self.config["DYNAMIC"]["LOGIC"]
        threshold = float(logic.get("CORRELATION_THRESHOLD", 0.0))
        window = int(self.config["DYNAMIC"].get("CORRELATION_WINDOW", 60))

        subset = prices[tickers].loc[:as_of].tail(window)
        returns = subset.pct_change().dropna()
        if returns.empty or len(tickers) <= 1:
            return tickers

        corr = returns.corr()
        avg_corr = corr.mean(axis=1)
        filtered = avg_corr[avg_corr >= threshold].index.tolist()
        return filtered if filtered else tickers

    def _ensure_fundamentals_loaded(self, tickers: List[str]) -> None:
        if (
            self._cached_market_cap is not None
            and self._cached_trading_value is not None
        ):
            return

        data_root = self._resolve_data_root()
        start_date = self._effective_start_date()
        end_date = self.config.get("END_DATE")

        print("[Universe] Loading Market Cap data...")
        self._cached_market_cap = load_local_parquet_field_bulk(
            tickers,
            ["market_cap", "marketcap", "MarketCap", "시가총액", "mkt_cap", "marcap"],
            data_root=str(data_root),
            start=start_date,
            end=end_date,
        )

        print("[Universe] Loading Trading Value data...")
        self._cached_trading_value = load_local_parquet_field_bulk(
            tickers,
            [
                "value",
                "trade_value",
                "trading_value",
                "거래대금",
                "amount",
                "turnover",
            ],
            data_root=str(data_root),
            start=start_date,
            end=end_date,
        )

    def filter_dynamic_candidates(
        self,
        tickers: List[str],
        close_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> List[str]:
        dynamic_cfg = self.config.get("DYNAMIC", {}) or {}
        sel_cfg = dynamic_cfg.get("SELECTION", {}) or {}
        filters_cfg = sel_cfg.get("FILTERS", {}) or {}

        min_age = int(filters_cfg.get("min_age_days", 750))
        min_price = float(filters_cfg.get("min_price", 1000))
        min_cap = float(filters_cfg.get("min_market_cap", 50_000_000_000))
        min_turnover = float(filters_cfg.get("min_turnover", 20_000_000_000))
        turnover_window = int(filters_cfg.get("turnover_window", 60))

        if not tickers:
            return []

        self._ensure_fundamentals_loaded(tickers)

        if as_of not in close_df.index:
            recent_idx = close_df.index[close_df.index <= as_of]
            if recent_idx.empty:
                return []
            current_idx = recent_idx[-1]
        else:
            current_idx = as_of

        current_prices = close_df.loc[current_idx]
        mask_price = (current_prices >= min_price) & (current_prices.notna())

        subset_hist = close_df.loc[:as_of]
        counts = subset_hist.count()
        mask_age = counts >= min_age

        mask_cap = pd.Series(False, index=close_df.columns)
        if self._cached_market_cap is not None and not self._cached_market_cap.empty:
            mc_idx = self._cached_market_cap.index[
                self._cached_market_cap.index <= as_of
            ]
            if not mc_idx.empty:
                current_mc = self._cached_market_cap.loc[mc_idx[-1]]
                current_mc = current_mc.reindex(close_df.columns, fill_value=0)
                mask_cap = current_mc >= min_cap

        mask_turnover = pd.Series(False, index=close_df.columns)
        if (
            self._cached_trading_value is not None
            and not self._cached_trading_value.empty
        ):
            tv_subset = self._cached_trading_value.loc[:as_of].tail(
                turnover_window + 10
            )
            if not tv_subset.empty:
                avg_tv = tv_subset.tail(turnover_window).mean()
                avg_tv = avg_tv.reindex(close_df.columns, fill_value=0)
                mask_turnover = avg_tv >= min_turnover
        else:
            pass

        final_mask = mask_price & mask_age & mask_cap & mask_turnover
        selected_tickers = final_mask[final_mask].index.tolist()

        return list(set(tickers) & set(selected_tickers))
