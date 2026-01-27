from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from core.engine import build_dynamic_components
from infra.logger import get_logger
from core.types import DynamicState


def _month_day_in_season(date: pd.Timestamp, start_md: str, end_md: str) -> bool:
    md = date.strftime("%m-%d")
    if start_md <= end_md:
        return start_md <= md <= end_md
    return md >= start_md or md <= end_md


def _freq_to_pandas(freq: str) -> str:
    mapping = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "ME",
        "Quarterly": "QE",
    }
    return mapping.get(freq, freq)


@dataclass
class LiveDynamicStrategy:
    config: Dict[str, object]

    def __post_init__(self) -> None:
        self.logger = get_logger("live.strategy.dynamic")
        config = self.config if isinstance(self.config, dict) else {}
        dynamic_cfg = config.get("DYNAMIC", {}) or {}
        if not isinstance(dynamic_cfg, dict):
            dynamic_cfg = {}
        self.logic = dynamic_cfg.get("LOGIC", {}) or {}
        if not isinstance(self.logic, dict):
            self.logic = {}
        self.rebalance_freq = str(dynamic_cfg.get("REBALANCE_FREQ", "Daily"))

        self.seasons = dynamic_cfg.get("SEASONS", []) or []

        self.selection_cfg = dynamic_cfg.get("SELECTION", {}) or {}
        if not isinstance(self.selection_cfg, dict):
            self.selection_cfg = {}
        self.selection_mode = str(self.selection_cfg.get("mode", "season")).lower()
        self.selection_top_n = int(self.selection_cfg.get("top_n", 10))
        self.selection_buy_n = int(
            self.selection_cfg.get("buy_n", self.selection_top_n)
        )
        if self.selection_buy_n <= 0:
            self.selection_buy_n = self.selection_top_n
        self.selection_buy_n = min(self.selection_buy_n, self.selection_top_n)

        self.selection_weighting = str(
            self.selection_cfg.get("weighting", "equal")
        ).lower()
        self.filters_cfg = self.selection_cfg.get("FILTERS", {}) or {}
        self.selection_corr_threshold = float(
            self.selection_cfg.get("corr_threshold", 0.3)
        )
        self.selection_corr_drop_pct = float(
            self.selection_cfg.get("corr_drop_pct", 0.0)
        )
        self.selection_momentum_window = int(dynamic_cfg.get("MOMENTUM_WINDOW", 60))
        self.selection_corr_window = int(dynamic_cfg.get("CORRELATION_WINDOW", 60))
        score_weights = self.selection_cfg.get("score_weights", {}) or {}
        self.selection_weight_corr = float(score_weights.get("corr", 0.4))
        self.selection_weight_mom = float(score_weights.get("momentum", 0.6))
        weight_sum = self.selection_weight_corr + self.selection_weight_mom
        if weight_sum <= 0:
            self.selection_weight_corr = 0.5
            self.selection_weight_mom = 0.5
        else:
            self.selection_weight_corr /= weight_sum
            self.selection_weight_mom /= weight_sum
        rank_weights = self.selection_cfg.get("rank_weights", []) or []
        if isinstance(rank_weights, list):
            self.selection_rank_weights = [float(w) for w in rank_weights]
        else:
            self.selection_rank_weights = []

        fee = dynamic_cfg.get("FEES", None)
        if fee is None:
            fee = config.get("FEES", 0.0)
        try:
            self.fee = float(fee)
        except (TypeError, ValueError):
            self.fee = 0.0

        self.slot_count = self._infer_slot_count()
        (
            self.correlation_filter,
            self.momentum_ranker,
            self.selector,
            self.allocator,
            self.stop_loss,
        ) = build_dynamic_components(
            self.config, fee=self.fee, slot_count=self.slot_count
        )

    def _infer_slot_count(self) -> int:
        alloc_cfg = self.logic.get("allocator", {}) or {}
        if isinstance(alloc_cfg, dict):
            slot_weights = alloc_cfg.get("slot_weights")
            if isinstance(slot_weights, list) and slot_weights:
                return len(slot_weights)

        legacy_weights = self.logic.get("SLOT_WEIGHTS")
        if isinstance(legacy_weights, list) and legacy_weights:
            return len(legacy_weights)

        return 4

    def rebalance_dates(self, index: pd.Index) -> set[pd.Timestamp]:
        freq = _freq_to_pandas(self.rebalance_freq)
        anchor = pd.Series(1, index=pd.DatetimeIndex(index))
        return set(anchor.resample(freq).last().index)

    def compute_targets(
        self,
        prices: pd.DataFrame,
        universe: List[str],
        as_of: pd.Timestamp,
        *,
        volumes: pd.DataFrame | None = None,
    ) -> Tuple[Dict[str, float], pd.Series, List[str]]:
        tickers = [t for t in universe if t in prices.columns]
        self.logger.info(
            "Compute targets: universe=%s in_prices=%s as_of=%s",
            len(universe),
            len(tickers),
            as_of.strftime("%Y-%m-%d"),
        )
        if not tickers:
            return {}, pd.Series(dtype=float), []

        tickers = self._filter_candidates(prices, volumes, tickers, as_of)
        self.logger.info("After filters: count=%s", len(tickers))
        if not tickers:
            return {}, pd.Series(dtype=float), []

        if self.selection_mode == "season" and self.seasons:
            season = self._current_season(as_of)
            season_tickers = season.get("tickers", []) or []
            if season_tickers:
                tickers = [t for t in tickers if t in season_tickers]
            self.logger.info(
                "Season filter: name=%s season_tickers=%s remaining=%s",
                season.get("name", "Season"),
                len(season_tickers),
                len(tickers),
            )
            if not tickers:
                return {}, pd.Series(dtype=float), []

        scores = self._score_candidates(prices, tickers, as_of)
        if scores.empty:
            return {}, pd.Series(dtype=float), []

        ranked = scores["total_score"].sort_values(ascending=False)
        self.logger.info("Scored candidates: count=%s", len(ranked))

        if self.selection_mode == "auto":
            top_ranked = ranked.head(self.selection_top_n)
            selected = list(top_ranked.index[: self.selection_buy_n])
            targets = self._auto_targets(selected, prices, as_of)
        else:
            selected = self.selector.select(ranked, self.slot_count)
            targets = self.allocator.targets(
                prices=prices, tickers=selected, as_of=as_of
            )
        return targets, ranked, selected

    def _score_candidates(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        window = int(self.selection_corr_window)
        windowed = prices[tickers].loc[:as_of].tail(window + 1)
        returns = windowed.pct_change(fill_method=None).dropna()
        if returns.empty:
            return pd.DataFrame()

        corr = returns.corr()
        corr_count = (corr >= self.selection_corr_threshold).sum(axis=1) - 1
        corr_count = corr_count.clip(lower=0).astype(float)

        if self.selection_corr_drop_pct > 0 and len(corr_count) > 1:
            drop_n = int(len(corr_count) * self.selection_corr_drop_pct)
            if drop_n > 0:
                lowest = corr_count.sort_values(ascending=True).iloc[:drop_n].index
                corr_count = corr_count.drop(lowest)
                tickers = [t for t in tickers if t in corr_count.index]
                if not tickers:
                    return pd.DataFrame()

        mom_window = int(self.selection_momentum_window)
        momentum = prices[tickers].loc[:as_of].pct_change(mom_window).iloc[-1]

        scores = pd.DataFrame(
            {"corr_count": corr_count, "momentum": momentum}, dtype="float64"
        ).dropna()
        if scores.empty:
            return pd.DataFrame()

        def _min_max(s: pd.Series) -> pd.Series:
            s = s.astype(float)
            min_v = float(s.min()) if not s.empty else 0.0
            max_v = float(s.max()) if not s.empty else 0.0
            if max_v == min_v:
                return pd.Series(0.0, index=s.index)
            return (s - min_v) / (max_v - min_v)

        scores["corr_score"] = _min_max(scores["corr_count"])
        scores["momentum_score"] = _min_max(scores["momentum"])
        scores["total_score"] = (
            scores["corr_score"] * self.selection_weight_corr
            + scores["momentum_score"] * self.selection_weight_mom
        )
        return scores.sort_values(
            ["total_score", "corr_score", "momentum_score"],
            ascending=[False, False, False],
        )

    def _auto_targets(
        self, selected: List[str], prices: pd.DataFrame, as_of: pd.Timestamp
    ) -> Dict[str, float]:
        if not selected:
            return {}

        if self.selection_weighting in {"rank", "rank_weights"}:
            weights = list(self.selection_rank_weights)[: len(selected)]
            if len(weights) < len(selected):
                weights += [0.0] * (len(selected) - len(weights))
            total = float(sum(weights))
            if total > 0:
                return {t: float(w) / total for t, w in zip(selected, weights)}

        if self.selection_weighting in {"allocator", "alloc"}:
            return self.allocator.targets(prices=prices, tickers=selected, as_of=as_of)

        w = 1.0 / float(len(selected))
        return {t: w for t in selected}

    def apply_stop_loss(
        self,
        state: DynamicState,
        prices: pd.Series,
        ranked: pd.Series,
        *,
        allowed_symbols: List[str] | None = None,
    ) -> DynamicState:
        if allowed_symbols is not None:
            allowed = set(allowed_symbols)
            holdings = {k: v for k, v in state.holdings.items() if k in allowed}
            high_water = {k: v for k, v in state.high_water.items() if k in allowed}
            state = DynamicState(
                cash=state.cash, holdings=holdings, high_water=high_water
            )
        return self.stop_loss.apply(state=state, prices=prices, ranked=ranked)

    def _current_season(self, date: pd.Timestamp) -> dict:
        for season in self.seasons:
            if _month_day_in_season(
                date, season.get("start_md", "01-01"), season.get("end_md", "12-31")
            ):
                return season
        return (
            self.seasons[0]
            if self.seasons
            else {
                "name": "Season",
                "start_md": "01-01",
                "end_md": "12-31",
                "tickers": [],
            }
        )

    def prefilter_candidates(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame | None,
        tickers: List[str],
        as_of: pd.Timestamp,
        *,
        min_age_override: int | None = None,
    ) -> List[str]:
        return self._filter_candidates(
            prices,
            volumes,
            tickers,
            as_of,
            min_age_override=min_age_override,
        )

    def _filter_candidates(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame | None,
        tickers: List[str],
        as_of: pd.Timestamp,
        *,
        min_age_override: int | None = None,
    ) -> List[str]:
        if not tickers:
            return []

        min_age = int(
            self.filters_cfg.get("min_age_days", 0)
            if min_age_override is None
            else min_age_override
        )
        min_price = float(self.filters_cfg.get("min_price", 0))
        min_turnover = float(self.filters_cfg.get("min_turnover", 0))
        turnover_window = int(self.filters_cfg.get("turnover_window", 20))

        if as_of not in prices.index:
            idx = prices.index[prices.index <= as_of]
            if idx.empty:
                return []
            current_idx = idx[-1]
        else:
            current_idx = as_of

        current_prices = prices.loc[current_idx].reindex(tickers)
        mask_price = (current_prices >= min_price) & current_prices.notna()

        mask_age = pd.Series(True, index=tickers)
        if min_age > 0:
            counts = prices.loc[:as_of, tickers].count()
            mask_age = counts >= min_age

        mask_turnover = pd.Series(True, index=tickers)
        if volumes is not None and min_turnover > 0:
            vol_subset = volumes.loc[:as_of, tickers].tail(turnover_window)
            px_subset = prices.loc[:as_of, tickers].tail(turnover_window)
            if not vol_subset.empty and not px_subset.empty:
                turnover = (vol_subset * px_subset).mean()
                mask_turnover = turnover >= min_turnover

        final = mask_price & mask_age & mask_turnover
        self.logger.info(
            "Filter stats: total=%s price_ok=%s age_ok=%s turnover_ok=%s final=%s min_age=%s min_price=%s min_turnover=%s window=%s",
            len(tickers),
            int(mask_price.sum()),
            int(mask_age.sum()),
            int(mask_turnover.sum()),
            int(final.sum()),
            min_age,
            min_price,
            min_turnover,
            turnover_window,
        )
        return [t for t in tickers if bool(final.get(t, False))]
