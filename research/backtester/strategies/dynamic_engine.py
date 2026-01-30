from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..universe import UniverseLoader
from core.engine import build_dynamic_components
from core.types import DynamicState
from ..utils import apply_fee, calc_nav, freq_to_pandas, month_day_in_season


@dataclass
class DynamicStrategy:
    config: Dict[str, Any]
    cash: float

    def __post_init__(self) -> None:
        dynamic_cfg = self.config.get("DYNAMIC", {}) or {}
        self.seasons = dynamic_cfg.get("SEASONS", []) or []
        self.logic = dynamic_cfg.get("LOGIC", {}) or {}
        self.rebalance_freq = str(dynamic_cfg.get("REBALANCE_FREQ", "Daily"))
        self.selection_cfg = dynamic_cfg.get("SELECTION", {}) or {}
        self.selection_mode = str(self.selection_cfg.get("mode", "season")).lower()
        self.selection_top_n = int(self.selection_cfg.get("top_n", 10))
        self.selection_buy_n = int(
            self.selection_cfg.get("buy_n", self.selection_top_n)
        )
        if self.selection_buy_n <= 0:
            self.selection_buy_n = self.selection_top_n
        self.selection_buy_n = min(self.selection_buy_n, self.selection_top_n)
        self.reserve_enabled = bool(self.selection_cfg.get("reserve_enabled", False))
        self.rebalance_timing = str(
            self.selection_cfg.get("rebalance_timing", "same_open")
        ).lower()
        reserve_order = self.selection_cfg.get("reserve_order", []) or []
        if isinstance(reserve_order, list):
            self.reserve_order = [int(x) for x in reserve_order if int(x) > 0]
        else:
            self.reserve_order = []
        self.reserve_from_rank = int(
            self.selection_cfg.get("reserve_from_rank", self.selection_buy_n + 1)
        )
        self.reserve_to_rank = int(
            self.selection_cfg.get("reserve_to_rank", self.selection_top_n)
        )
        self.selection_lookback_years = int(self.selection_cfg.get("lookback_years", 3))
        self.selection_corr_threshold = float(
            self.selection_cfg.get("corr_threshold", 0.3)
        )
        self.selection_corr_drop_pct = float(
            self.selection_cfg.get("corr_drop_pct", 0.2)
        )
        self.selection_momentum_window = int(dynamic_cfg.get("MOMENTUM_WINDOW", 60))
        self.selection_corr_window = int(dynamic_cfg.get("CORRELATION_WINDOW", 60))
        self.selection_weighting = str(
            self.selection_cfg.get("weighting", "equal")
        ).lower()
        rank_weights = self.selection_cfg.get("rank_weights", []) or []
        if isinstance(rank_weights, list):
            self.selection_rank_weights = [float(w) for w in rank_weights]
        else:
            self.selection_rank_weights = []
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
        fee = dynamic_cfg.get("FEES", None)
        if fee is None:
            fee = self.config.get("FEES", 0.0)
        self.fee = float(fee)

        self.holdings: Dict[str, float] = {}
        self.high_water: Dict[str, float] = {}
        self.active_season: Optional[str] = None
        self.active_period: Optional[str] = None
        self.pending_shares: Dict[str, float] | None = None
        self.pending_targets: Dict[str, float] | None = None
        self.prev_close_prices: Optional[pd.Series] = None
        self.prev_date: Optional[pd.Timestamp] = None
        self.selection_log: List[Dict[str, Any]] = []
        self.last_ranked: Optional[pd.Series] = None
        self.reserve_list: List[str] = []

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

    def rebalance_dates(self, index: pd.Index) -> set[pd.Timestamp]:
        freq = freq_to_pandas(self.rebalance_freq)
        anchor = pd.Series(1, index=pd.DatetimeIndex(index))
        return set(anchor.resample(freq).last().index)

    def _current_season(self, date: pd.Timestamp) -> dict:
        if not self.seasons:
            return {
                "name": "Season",
                "start_md": "01-01",
                "end_md": "12-31",
                "tickers": [],
            }
        for season in self.seasons:
            if month_day_in_season(date, season["start_md"], season["end_md"]):
                return season
        return self.seasons[0]

    def _get_target_season(
        self, date: pd.Timestamp, *, is_rebalancing_day: bool
    ) -> dict:
        if self.selection_mode != "auto":
            return self._current_season(date)
        if not is_rebalancing_day:
            return self._current_season(date)
        next_day = date + pd.Timedelta(days=1)
        return self._current_season(next_day)

    def _build_reserve_list(self, scores: pd.DataFrame) -> List[str]:
        if scores is None or scores.empty:
            return []
        ordered = list(scores.index)
        if self.reserve_order:
            result: list[str] = []
            for r in self.reserve_order:
                if 1 <= r <= len(ordered):
                    ticker = ordered[r - 1]
                    if ticker not in result:
                        result.append(ticker)
            return result
        start_rank = max(1, int(self.reserve_from_rank))
        end_rank = max(start_rank, int(self.reserve_to_rank))
        return ordered[start_rank - 1 : end_rank]

    def _fill_from_reserve(self, prices_df: pd.DataFrame, as_of: pd.Timestamp) -> None:
        if not self.reserve_enabled or not self.reserve_list:
            return
        if self.selection_buy_n <= 0:
            return
        current = set(self.holdings.keys())
        if len(current) >= self.selection_buy_n:
            return

        prices = (
            prices_df.loc[as_of] if as_of in prices_df.index else prices_df.iloc[-1]
        )
        nav = calc_nav(prices, self.holdings, self.cash)
        if nav <= 0 or self.cash <= 0:
            return

        candidates = [t for t in self.reserve_list if t not in current]
        if not candidates:
            return

        # allocate available cash to next reserve candidate
        next_ticker = candidates[0]
        all_tickers = list(current) + [next_ticker]
        weights = self._auto_targets(all_tickers, prices_df, as_of)
        target_weight = float(weights.get(next_ticker, 0.0))
        if target_weight <= 0:
            return

        target_value = nav * target_weight
        invest = min(self.cash, target_value)
        px = float(prices.get(next_ticker, 0.0))
        if px <= 0 or pd.isna(px):
            return
        shares = invest / px
        if shares <= 0:
            return
        trade_value = shares * px
        fee = apply_fee(trade_value, self.fee)
        self.cash -= trade_value + fee
        self.holdings[next_ticker] = self.holdings.get(next_ticker, 0.0) + shares
        self.high_water[next_ticker] = max(self.high_water.get(next_ticker, px), px)

    def _season_year(self, date: pd.Timestamp, season: dict) -> int:
        start_md = str(season.get("start_md", "01-01"))
        end_md = str(season.get("end_md", "12-31"))
        start_mmdd = tuple(int(x) for x in start_md.split("-"))
        end_mmdd = tuple(int(x) for x in end_md.split("-"))
        if start_mmdd <= end_mmdd:
            return date.year
        if (date.month, date.day) <= end_mmdd:
            return date.year - 1
        return date.year

    def _season_window(
        self, season_year: int, season: dict
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_md = str(season.get("start_md", "01-01"))
        end_md = str(season.get("end_md", "12-31"))
        start_mm, start_dd = (int(x) for x in start_md.split("-"))
        end_mm, end_dd = (int(x) for x in end_md.split("-"))
        start = pd.Timestamp(season_year, start_mm, start_dd)
        if (start_mm, start_dd) <= (end_mm, end_dd):
            end = pd.Timestamp(season_year, end_mm, end_dd)
        else:
            end = pd.Timestamp(season_year + 1, end_mm, end_dd)
        return start, end

    def _season_key(self, season_year: int, season: dict) -> str:
        name = str(season.get("name", "Season"))
        return f"{season_year}-{name}"

    def _eligible_season_winners(
        self,
        tickers: List[str],
        close_df: pd.DataFrame,
        open_df: pd.DataFrame,
        season: dict,
        season_year: int,
    ) -> List[str]:
        winners: List[str] = []

        for t in tickers:
            ok = True
            for y in range(
                season_year, season_year - self.selection_lookback_years, -1
            ):
                start, end = self._season_window(y, season)
                closes = close_df.get(t)
                opens = open_df.get(t)
                if closes is None or opens is None:
                    ok = False
                    break

                close_slice = closes.loc[start:end]
                open_slice = opens.loc[start:end]
                close_slice = close_slice.dropna()
                open_slice = open_slice.dropna()
                if close_slice.empty or open_slice.empty:
                    ok = False
                    break

                entry_open = float(open_slice.iloc[0])
                exit_close = float(close_slice.iloc[-1])
                if (
                    not pd.notna(entry_open)
                    or not pd.notna(exit_close)
                    or entry_open <= 0
                ):
                    ok = False
                    break
                if (exit_close / entry_open) - 1.0 <= 0:
                    ok = False
                    break

            if ok:
                winners.append(t)

        return winners

    def _season_returns_concat(
        self,
        tickers: List[str],
        close_df: pd.DataFrame,
        season: dict,
        season_year: int,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for y in range(season_year, season_year - self.selection_lookback_years, -1):
            start, end = self._season_window(y, season)
            sub = close_df[tickers].loc[
                (close_df.index >= start) & (close_df.index <= end)
            ]
            returns = sub.pct_change(fill_method=None).dropna()
            if not returns.empty:
                frames.append(returns)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=0, ignore_index=True)

    def _score_candidates(
        self,
        tickers: List[str],
        close_df: pd.DataFrame,
        season: dict,
        season_year: int,
        as_of: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not tickers:
            return pd.DataFrame(), pd.DataFrame()

        returns = self._season_returns_concat(tickers, close_df, season, season_year)
        if returns.empty:
            return pd.DataFrame(), pd.DataFrame()

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
                    return pd.DataFrame(), corr

        mom_window = int(self.selection_momentum_window)
        momentum = close_df[tickers].loc[:as_of].pct_change(mom_window).iloc[-1]

        scores = pd.DataFrame({"corr_count": corr_count, "momentum": momentum}).dropna()
        if scores.empty:
            return pd.DataFrame(), corr

        scores = scores.assign(ticker=scores.index.astype(str))

        def _min_max(s: pd.Series) -> pd.Series:
            s = s.astype(float)
            min_v = float(s.min()) if not s.empty else 0.0
            max_v = float(s.max()) if not s.empty else 0.0
            if max_v == min_v:
                return pd.Series(0.0, index=s.index)
            return (s - min_v) / (max_v - min_v)

        scores["corr_score"] = _min_max(scores["corr_count"])
        scores["momentum_score"] = _min_max(scores["momentum"])

        corr_order = scores.sort_values(
            ["corr_score", "momentum_score", "ticker"],
            ascending=[False, False, True],
        )
        scores.loc[corr_order.index, "corr_rank"] = range(1, len(corr_order) + 1)

        mom_order = scores.sort_values(
            ["momentum_score", "corr_score", "ticker"],
            ascending=[False, False, True],
        )
        scores.loc[mom_order.index, "momentum_rank"] = range(1, len(mom_order) + 1)

        scores["corr_rank"] = scores["corr_rank"].astype(float)
        scores["momentum_rank"] = scores["momentum_rank"].astype(float)
        scores["total_score"] = (
            scores["corr_score"] * self.selection_weight_corr
            + scores["momentum_score"] * self.selection_weight_mom
        )
        scores = scores.sort_values(
            ["total_score", "corr_score", "momentum_score", "ticker"],
            ascending=[False, False, False, True],
        )
        scores = scores.drop(columns=["ticker", "corr_score", "momentum_score"])
        return scores, corr

    def _auto_targets(
        self,
        selected: List[str],
        prices_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> Dict[str, float]:
        if not selected:
            return {}
        if self.selection_weighting == "allocator":
            return self.allocator.targets(
                prices=prices_df, tickers=selected, as_of=as_of
            )
        if self.selection_weighting in {"rank", "rank_weights", "custom"}:
            if self.selection_rank_weights:
                weights = list(self.selection_rank_weights)[: len(selected)]
                if len(weights) < len(selected):
                    weights += [0.0] * (len(selected) - len(weights))
                total = float(sum(weights))
                if total > 0:
                    return {t: float(w) / total for t, w in zip(selected, weights)}

        weight = 1.0 / float(len(selected))
        return {t: weight for t in selected}

    def _record_selection(
        self,
        date: pd.Timestamp,
        season_key: str,
        selected: List[str],
        scores: pd.DataFrame,
        corr: pd.DataFrame,
        *,
        season: dict,
        season_year: int,
        total_universe: int,
        eligible_universe: int,
    ) -> None:
        corr_count = scores["corr_count"].to_dict()
        momentum = scores["momentum"].to_dict()
        corr_rank = scores["corr_rank"].to_dict()
        momentum_rank = scores["momentum_rank"].to_dict()
        total_score = scores["total_score"].to_dict()
        corr_matrix = {}
        if not corr.empty and selected:
            sub = corr.loc[selected, selected].round(6)
            corr_matrix = {
                t: {k: float(v) for k, v in sub.loc[t].items()} for t in sub.index
            }

        self.selection_log.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "season": season_key,
                "season_year": int(season_year),
                "season_name": str(season.get("name", "Season")),
                "season_start": str(season.get("start_md", "01-01")),
                "season_end": str(season.get("end_md", "12-31")),
                "universe_count": int(total_universe),
                "eligible_count": int(eligible_universe),
                "scored_count": int(scores.shape[0]),
                "selected": selected,
                "corr_count": {k: float(v) for k, v in corr_count.items()},
                "momentum": {k: float(v) for k, v in momentum.items()},
                "corr_rank": {k: float(v) for k, v in corr_rank.items()},
                "momentum_rank": {k: float(v) for k, v in momentum_rank.items()},
                "total_score": {k: float(v) for k, v in total_score.items()},
                "corr_matrix": corr_matrix,
            }
        )

    def _infer_slot_count(self) -> int:
        # 플러그인 스키마
        alloc_cfg = self.logic.get("allocator", {}) or {}
        if isinstance(alloc_cfg, dict):
            slot_weights = alloc_cfg.get("slot_weights")
            if isinstance(slot_weights, list) and slot_weights:
                return len(slot_weights)

        # 레거시 스키마
        legacy_weights = self.logic.get("SLOT_WEIGHTS")
        if isinstance(legacy_weights, list) and legacy_weights:
            return len(legacy_weights)

        return 4

    def _calc_target_shares(
        self, prices: pd.Series, targets: Dict[str, float]
    ) -> Dict[str, float]:
        nav = calc_nav(prices, self.holdings, self.cash)
        target_shares: Dict[str, float] = {}

        universe = set(self.holdings.keys()) | set(targets.keys())
        for t in universe:
            price = float(prices.get(t, 0.0))
            if price <= 0 or pd.isna(price):
                target_shares[t] = self.holdings.get(t, 0.0)
                continue
            w = float(targets.get(t, 0.0))
            target_value = nav * w
            target_shares[t] = target_value / price
        return target_shares

    def _execute_to_shares(
        self, prices: pd.Series, target_shares: Dict[str, float]
    ) -> Dict[str, float]:
        remaining: Dict[str, float] = {}
        for t, target in target_shares.items():
            price = float(prices.get(t, 0.0))
            if price <= 0 or pd.isna(price):
                remaining[t] = target
                continue
            current = self.holdings.get(t, 0.0)
            delta = target - current
            trade_value = delta * price
            fee = apply_fee(trade_value, self.fee)
            self.cash -= trade_value + fee
            new_shares = current + delta
            if new_shares <= 0:
                self.holdings.pop(t, None)
                self.high_water.pop(t, None)
            else:
                self.holdings[t] = new_shares
                self.high_water[t] = max(self.high_water.get(t, price), price)
        return remaining

    def _liquidate_all(self, prices: pd.Series) -> None:
        for t in list(self.holdings.keys()):
            price = float(prices.get(t, 0.0))
            if price <= 0 or pd.isna(price):
                continue
            shares = self.holdings.get(t, 0.0)
            if shares <= 0:
                continue
            trade_value = shares * price
            fee = apply_fee(trade_value, self.fee)
            self.cash += trade_value - fee
            self.holdings.pop(t, None)
            self.high_water.pop(t, None)

    def on_day(
        self,
        date: pd.Timestamp,
        close_prices: pd.Series,
        open_prices: pd.Series,
        prices_df: pd.DataFrame,
        open_prices_df: pd.DataFrame,
        rebalance_set: set[pd.Timestamp],
        loader: UniverseLoader,
    ) -> None:
        as_of = self.prev_date or date

        if self.pending_targets is not None:
            target_shares = self._calc_target_shares(open_prices, self.pending_targets)
            remaining = self._execute_to_shares(open_prices, target_shares)
            self.pending_shares = remaining or None
            self.pending_targets = None

        if self.pending_shares:
            remaining = self._execute_to_shares(open_prices, self.pending_shares)
            self.pending_shares = remaining or None

        if self.selection_mode == "auto":
            current_season = self._current_season(date)
            current_year = self._season_year(date, current_season)
            current_key = self._season_key(current_year, current_season)
            is_rebalancing_day = (
                self.active_period != current_key or date in rebalance_set
            )
            if is_rebalancing_day:
                self.active_period = current_key

            if date in rebalance_set:
                as_of = date

            season = self._get_target_season(
                date, is_rebalancing_day=is_rebalancing_day
            )
            season_ref_date = (
                date + pd.Timedelta(days=1) if is_rebalancing_day else date
            )
            season_year = self._season_year(season_ref_date, season)
            season_key = self._season_key(season_year, season)

            tickers = loader.dynamic_universe_tickers()
            if not tickers:
                return

            scores = pd.DataFrame()
            corr = pd.DataFrame()
            eligible: List[str] = []

            if is_rebalancing_day:
                tickers = loader.filter_dynamic_candidates(tickers, prices_df, as_of)
                if not tickers:
                    return

                eligible = self._eligible_season_winners(
                    tickers, prices_df, open_prices_df, season, season_year
                )
                scores, corr = self._score_candidates(
                    eligible, prices_df, season, season_year, as_of
                )
                if scores.empty:
                    return

                ranked = scores["momentum"]
                self.last_ranked = ranked
                self.reserve_list = self._build_reserve_list(scores)
            else:
                if self.last_ranked is None:
                    eligible = self._eligible_season_winners(
                        tickers, prices_df, open_prices_df, season, season_year
                    )
                    scores, corr = self._score_candidates(
                        eligible, prices_df, season, season_year, as_of
                    )
                    if scores.empty:
                        return
                    ranked = scores["momentum"]
                    self.last_ranked = ranked
                    self.reserve_list = self._build_reserve_list(scores)
                else:
                    ranked = self.last_ranked
        else:
            season = self._current_season(date)
            is_rebalancing_day = self.active_season != season["name"]
            if is_rebalancing_day:
                self.active_season = season["name"]

            tickers = season.get("tickers", []) or []
            if not tickers:
                return

            filtered = self.correlation_filter.filter(prices_df, tickers, as_of)
            ranked = self.momentum_ranker.rank(prices_df, filtered, as_of)
            if ranked.empty:
                return
            self.last_ranked = ranked

        if is_rebalancing_day:
            if self.selection_mode == "auto":
                selected = list(scores.head(self.selection_buy_n).index)
                targets = self._auto_targets(selected, prices_df, date)
                if self.rebalance_timing == "next_open" and date in rebalance_set:
                    self._liquidate_all(close_prices)
                    self.pending_targets = targets
                else:
                    if self.prev_close_prices is not None:
                        self._liquidate_all(self.prev_close_prices)
                    target_shares = self._calc_target_shares(open_prices, targets)
                    remaining = self._execute_to_shares(open_prices, target_shares)
                    self.pending_shares = remaining or None
                self._record_selection(
                    date,
                    season_key,
                    selected,
                    scores,
                    corr,
                    season=season,
                    season_year=season_year,
                    total_universe=len(tickers),
                    eligible_universe=len(eligible),
                )
            else:
                selected = self.selector.select(ranked, self.slot_count)
                targets = self.allocator.targets(
                    prices=prices_df, tickers=selected, as_of=date
                )
                if self.rebalance_timing == "next_open" and date in rebalance_set:
                    self._liquidate_all(close_prices)
                    self.pending_targets = targets
                else:
                    if self.prev_close_prices is not None:
                        self._liquidate_all(self.prev_close_prices)
                    target_shares = self._calc_target_shares(open_prices, targets)
                    remaining = self._execute_to_shares(open_prices, target_shares)
                    self.pending_shares = remaining or None

        state = DynamicState(
            cash=self.cash, holdings=self.holdings, high_water=self.high_water
        )
        state = self.stop_loss.apply(state=state, prices=close_prices, ranked=ranked)
        self.cash, self.holdings, self.high_water = (
            state.cash,
            state.holdings,
            state.high_water,
        )

        if self.selection_mode == "auto" and self.reserve_enabled:
            self._fill_from_reserve(prices_df, date)

        self.prev_close_prices = close_prices
        self.prev_date = date

    def nav(self, prices: pd.Series) -> float:
        return calc_nav(prices, self.holdings, self.cash)

    def add_cash(self, amount: float) -> None:
        self.cash += amount

    def withdraw_cash(self, amount: float, prices: pd.Series) -> None:
        if amount <= 0:
            return
        nav = self.nav(prices)
        if nav <= 0:
            return
        ratio = min(amount / nav, 1.0)
        for t in list(self.holdings.keys()):
            price = float(prices.get(t, 0.0))
            if price <= 0 or pd.isna(price):
                continue
            sell_shares = self.holdings[t] * ratio
            trade_value = sell_shares * price
            fee = apply_fee(trade_value, self.fee)
            self.cash += trade_value - fee
            self.holdings[t] -= sell_shares
