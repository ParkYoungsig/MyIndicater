from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


def _configure_plot_fonts() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Glyph .* missing from font",
        category=UserWarning,
    )
    if sys.platform.startswith("win"):
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


if __package__ is None or __package__ == "":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from research.backtester.runner import run_dual_engine_backtest
    from research.backtester.config import load_yaml_config, resolve_logic_config_name
    from research.backtester.metrics import performance_summary
    from research.backtester.utils import month_day_in_season
else:
    from .runner import run_dual_engine_backtest
    from .config import load_yaml_config, resolve_logic_config_name
    from .metrics import performance_summary
    from .utils import month_day_in_season


def main() -> None:
    _configure_plot_fonts()
    parser = argparse.ArgumentParser(description="Backtester")
    parser.add_argument(
        "--config", type=str, default="backtester", help="config/{name}.yaml"
    )
    parser.add_argument(
        "--logic",
        type=str,
        default=None,
        help="logic config name (e.g. logic_fast_follower_01 or config.logic_fast_follower_01)",
    )
    parser.add_argument(
        "--json-run",
        type=str,
        default=None,
        help="results run folder name to load config_backtester.json & config_logic.json",
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--print-equity-head", type=int, default=0)
    args = parser.parse_args()

    result = run_dual_engine_backtest(
        config_name=args.config,
        logic_name=args.logic,
        json_run=args.json_run,
        start=args.start,
        end=args.end,
    )
    print("[Backtester]", result.performance)
    print("[Benchmark]", result.benchmark_performance)

    _save_results(result, args.config, args.logic)

    if args.print_equity_head and not result.equity.empty:
        print(result.equity.head(args.print_equity_head))


def _resolve_output_root(cfg: dict) -> Path:
    results_cfg = cfg.get("RESULTS", {}) or {}
    root = results_cfg.get("root") or "results"
    root_path = Path(root)
    if not root_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        root_path = (project_root / root_path).resolve()
    return root_path


def _resolve_config_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / path).resolve()


def _drawdown_series(equity: pd.Series | None) -> pd.Series:
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _mdd_from_equity(equity: pd.Series) -> float:
    if equity is None or equity.empty:
        return 0.0
    dd = _drawdown_series(equity)
    return float(dd.min()) if not dd.empty else 0.0


def _perf_dict(equity: pd.Series, returns: pd.Series, trading_days: int) -> dict:
    if equity is None or equity.empty:
        return {"CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "TotalReturn": 0.0}
    return performance_summary(equity, returns, trading_days)


def _format_metrics(metrics: dict) -> dict:
    return {
        "CAGR": float(metrics.get("CAGR", 0.0)),
        "MDD": float(metrics.get("MDD", 0.0)),
        "Sharpe": float(metrics.get("Sharpe", 0.0)),
        "TotalReturn": float(metrics.get("TotalReturn", 0.0)),
    }


def _round_numeric_df(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        return df
    df.loc[:, num_cols] = df.loc[:, num_cols].round(decimals)
    return df


def _format_ref_years(season_year: Any, lookback_years: int) -> str:
    try:
        year = int(season_year)
    except (TypeError, ValueError):
        return ""
    if lookback_years <= 1:
        return str(year)
    start = year - lookback_years + 1
    return f"{start}-{year}"


def _season_mask(index: pd.DatetimeIndex, start_md: str, end_md: str) -> pd.Series:
    return pd.Series(
        [month_day_in_season(pd.Timestamp(d), start_md, end_md) for d in index],
        index=index,
    )


def _load_sector_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    code_col = None
    for key in ("code", "ticker", "symbol"):
        if key in cols:
            code_col = cols[key]
            break
    sector_col = None
    for key in ("sector", "industry", "category"):
        if key in cols:
            sector_col = cols[key]
            break
    if code_col is None or sector_col is None:
        return {}

    codes = df[code_col].astype(str).str.replace(".0", "", regex=False).str.strip()
    codes = codes.str.zfill(6)
    sectors = df[sector_col].astype(str).str.strip()
    mapping = dict(zip(codes, sectors))
    return {k: v for k, v in mapping.items() if v and v != "nan"}


def _load_name_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    code_col = None
    for key in ("code", "ticker", "symbol"):
        if key in cols:
            code_col = cols[key]
            break
    name_col = None
    for key in ("name", "종목명", "회사명", "company", "company_name"):
        if key.lower() in cols:
            name_col = cols[key.lower()]
            break
    if code_col is None or name_col is None:
        return {}

    codes = df[code_col].astype(str).str.replace(".0", "", regex=False).str.strip()
    codes = codes.str.zfill(6)
    names = df[name_col].astype(str).str.strip()
    mapping = dict(zip(codes, names))
    return {k: v for k, v in mapping.items() if v and v != "nan"}


def _season_window(season_year: int, season: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
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


def _build_monthly_heatmap(series: pd.Series, years: int = 10) -> pd.DataFrame:
    if series is None or series.empty:
        return pd.DataFrame()
    monthly = series.resample("ME").last().pct_change(fill_method=None).dropna()
    if monthly.empty:
        return pd.DataFrame()
    df = monthly.to_frame("ret")
    idx = pd.DatetimeIndex(df.index)
    df["year"] = idx.year
    df["month"] = idx.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    if years > 0:
        pivot = pivot.loc[pivot.index.sort_values().tolist()[-years:]]
    return pivot


def _build_season_heatmap(
    series: pd.Series, seasons: list[dict], years: int = 10
) -> pd.DataFrame:
    if series is None or series.empty or not seasons:
        return pd.DataFrame()
    years_all = sorted(set(pd.DatetimeIndex(series.index).year))
    if years > 0:
        years_all = years_all[-years:]

    rows: list[dict[str, Any]] = []
    for y in years_all:
        row: dict[str, Any] = {"year": y}
        for season in seasons:
            name = str(season.get("name", "Season"))
            start, end = _season_window(y, season)
            sub = series.loc[(series.index >= start) & (series.index <= end)]
            if sub.empty:
                row[name] = np.nan
            else:
                row[name] = float(sub.iloc[-1] / sub.iloc[0] - 1.0)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("year")
    return df


def _plot_heatmap(df: pd.DataFrame, title: str, path: Path) -> None:
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    data = df.values.astype(float)
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn")
    ax.set_title(title)
    ax.set_xlabel("Month/Season")
    ax.set_ylabel("Year")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(list(df.columns))
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels([str(y) for y in df.index])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cash_weights(
    total_cash: pd.Series,
    dynamic_cash: pd.Series,
    static_cash: pd.Series,
    path: Path,
) -> None:
    if total_cash is None or total_cash.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(total_cash.index, total_cash.to_numpy(), label="Total Cash")
    if dynamic_cash is not None and not dynamic_cash.empty:
        ax.plot(dynamic_cash.index, dynamic_cash.to_numpy(), label="Dynamic Cash")
    if static_cash is not None and not static_cash.empty:
        ax.plot(static_cash.index, static_cash.to_numpy(), label="Static Cash")
    ax.set_title("Cash Weight Dynamics")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cash Weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_underwater(drawdown: pd.Series, path: Path) -> None:
    if drawdown is None or drawdown.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(drawdown.index, drawdown.to_numpy(), 0, color="#d73027")
    ax.set_title("Underwater Plot")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_zoom(
    equity: pd.Series,
    benchmark: pd.Series,
    cash_weight: pd.Series,
    start: str,
    end: str,
    title: str,
    path: Path,
) -> None:
    if equity is None or equity.empty:
        return
    sub_eq = equity.loc[start:end]
    if sub_eq.empty:
        return
    sub_eq = sub_eq / sub_eq.iloc[0]
    sub_bench = None
    if benchmark is not None and not benchmark.empty:
        sub_bench = benchmark.loc[start:end]
        if not sub_bench.empty:
            sub_bench = sub_bench / sub_bench.iloc[0]
    sub_cash = None
    if cash_weight is not None and not cash_weight.empty:
        sub_cash = cash_weight.loc[start:end]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sub_eq.index, sub_eq.values, label="Strategy")
    if sub_bench is not None and not sub_bench.empty:
        ax.plot(sub_bench.index, sub_bench.to_numpy(), label="Benchmark")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Equity")
    ax.legend(loc="upper left")

    if sub_cash is not None and not sub_cash.empty:
        ax2 = ax.twinx()
        ax2.plot(
            sub_cash.index, sub_cash.to_numpy(), color="gray", alpha=0.6, label="Cash"
        )
        ax2.set_ylabel("Cash Weight")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    boxes = [
        (0.05, 0.6, "Data Loader"),
        (0.3, 0.6, "Universe Filter"),
        (0.55, 0.6, "Dual Engine\n(Master/Dynamic)"),
        (0.8, 0.6, "Executor (HTS)"),
    ]
    for x, y, label in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            0.18,
            0.25,
            boxstyle="round,pad=0.02",
            edgecolor="#333",
            facecolor="#f2f2f2",
        )
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.125, label, ha="center", va="center", fontsize=9)

    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.18
        y1 = boxes[i][1] + 0.125
        x2 = boxes[i + 1][0]
        y2 = boxes[i + 1][1] + 0.125
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->"})

    ax.set_title("System Architecture")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _regime_analysis(
    benchmark_returns: pd.Series, strategy_returns: pd.Series
) -> pd.DataFrame:
    if benchmark_returns is None or benchmark_returns.empty:
        return pd.DataFrame()
    r = benchmark_returns.dropna()
    if r.empty:
        return pd.DataFrame()

    rolling = (1 + r).rolling(126).apply(lambda x: np.prod(x) - 1, raw=False)
    regime = pd.Series(index=r.index, dtype=object)
    regime[rolling > 0.05] = "Bull"
    regime[rolling < -0.05] = "Bear"
    regime[(rolling >= -0.05) & (rolling <= 0.05)] = "Sideways"
    regime = regime.dropna()

    aligned = pd.DataFrame({"regime": regime, "strategy": strategy_returns}).dropna()
    if aligned.empty:
        return pd.DataFrame()
    summary = (
        aligned.groupby("regime")["strategy"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "days", "mean": "avg_daily_return", "std": "vol"})
    )
    summary["share"] = summary["days"] / summary["days"].sum()
    return summary.reset_index()




def _copy_artifacts_to_category_folders(run_dir: Path) -> None:
    """Create convenience subfolders and copy root-level artifacts into them.

    Policy (Direction B):
    - DO NOT move or rename existing root artifacts.
    - Always create categorized folders and place COPIES of artifacts for human convenience.
    """
    folders = {
        "charts": run_dir / "charts",
        "tables": run_dir / "tables",
        "logs": run_dir / "logs",
        "configs": run_dir / "configs",
        "report": run_dir / "report",
    }
    for p in folders.values():
        p.mkdir(parents=True, exist_ok=True)

    config_names = {"config_backtester.json", "config_logic.json", "config_merged.json"}
    report_names = {"summary.md"}

    for item in run_dir.iterdir():
        if not item.is_file():
            continue
        name = item.name
        lower = name.lower()

        if lower.endswith(".png"):
            dst_dir = folders["charts"]
        elif lower.endswith(".csv"):
            dst_dir = folders["tables"]
        elif name in config_names:
            dst_dir = folders["configs"]
        elif lower.endswith(".json"):
            dst_dir = folders["logs"]
        elif name in report_names:
            dst_dir = folders["report"]
        else:
            continue

        shutil.copy2(item, dst_dir / name)


def _replay_total_equity_without_stoploss(result: "DualEngineResult") -> "pd.Series":
    """Recompute TOTAL equity assuming stop-loss is disabled, without re-running the backtest.

    This avoids a second call to run_dual_engine_backtest() (and thus avoids re-loading Universe data).

    We keep the static sleeve equity exactly as in the original run, and re-simulate only the dynamic sleeve
    from the selection log using equal-weight allocation. This preserves the mdd_defense artifacts while
    ensuring the backtest runs only once.
    """
    prices = result.prices
    if prices is None or prices.empty:
        return result.equity.copy()

    idx = pd.DatetimeIndex(prices.index)
    prices = prices.copy()
    prices.index = idx

    dyn_equity0 = float(result.dynamic_equity.iloc[0]) if len(result.dynamic_equity) else 0.0

    cfg = result.config or {}
    dyn_cfg = cfg.get("DYNAMIC", {}) or {}
    sel_cfg = dyn_cfg.get("SELECTION", {}) or {}
    timing = str(sel_cfg.get("rebalance_timing", "next_open")).lower()
    fee = float(dyn_cfg.get("FEES", 0.0) or 0.0)

    sel_map = {}
    for rec in (result.dynamic_selection_log or []):
        if "date" not in rec:
            continue
        d = pd.Timestamp(rec["date"])
        sel_map[d] = list(rec.get("selected") or [])

    index_list = list(prices.index)
    index_pos = {d: i for i, d in enumerate(index_list)}

    def trade_date(selection_date: pd.Timestamp):
        if selection_date not in index_pos:
            return None
        if timing == "same_open":
            return selection_date
        i = index_pos[selection_date]
        if i + 1 >= len(index_list):
            return None
        return index_list[i + 1]

    trade_map = {}
    for sd, selected in sel_map.items():
        td = trade_date(sd)
        if td is not None:
            trade_map[td] = selected

    cash = dyn_equity0
    shares: dict[str, float] = {}
    dyn_equity = pd.Series(index=prices.index, dtype="float64")

    def portfolio_value(row) -> float:
        v = cash
        for t, sh in shares.items():
            px = row.get(t)
            if pd.notna(px):
                v += float(sh) * float(px)
        return float(v)

    for d in prices.index:
        row = prices.loc[d]

        if d in trade_map:
            selected = trade_map[d]

            sell_notional = 0.0
            for t, sh in list(shares.items()):
                px = row.get(t)
                if pd.isna(px):
                    continue
                notional = float(sh) * float(px)
                sell_notional += notional
                cash += notional
                shares.pop(t, None)
            if fee > 0 and sell_notional > 0:
                cash -= sell_notional * fee

            selected = [t for t in selected if pd.notna(row.get(t))]
            if selected and cash > 0:
                target_each = cash / float(len(selected))
                buy_notional = 0.0
                for t in selected:
                    px = float(row.get(t))
                    if px <= 0:
                        continue
                    sh = target_each / px
                    shares[t] = shares.get(t, 0.0) + sh
                    buy_notional += sh * px
                cash -= buy_notional
                if fee > 0 and buy_notional > 0:
                    cash -= buy_notional * fee

        dyn_equity.loc[d] = portfolio_value(row)

    static_eq = result.static_equity.reindex(dyn_equity.index).ffill()
    return static_eq + dyn_equity




def _normalize_selection_log_for_json(selection_log):
    """Normalize selection log for deterministic JSON output (no effect on backtest results).
    - Sort ticker lists inside each record when present.
    """
    if not isinstance(selection_log, list):
        return selection_log
    out = []
    for rec in selection_log:
        if not isinstance(rec, dict):
            out.append(rec)
            continue
        r = dict(rec)
        for k in ("selected", "candidates", "dropped", "reserve", "removed"):
            v = r.get(k)
            if isinstance(v, list):
                try:
                    r[k] = sorted(v)
                except Exception:
                    pass
        out.append(r)
    return out

def _save_results(result, config_name: str, logic_name: str | None) -> None:
    report_start = time.perf_counter()
    output_root = _resolve_output_root(result.config)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{config_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        user_cfg = load_yaml_config(config_name)
    except Exception:
        user_cfg = {}

    logic_cfg = {}
    logic_config_name = resolve_logic_config_name(user_cfg, logic_name)
    if logic_config_name:
        try:
            logic_cfg = load_yaml_config(logic_config_name)
        except Exception:
            logic_cfg = {}

    (run_dir / "config_backtester.json").write_text(
        json.dumps(user_cfg, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "config_logic.json").write_text(
        json.dumps(logic_cfg, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "config_merged.json").write_text(
        json.dumps(result.config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    sector_map = (
        result.config.get("SECTOR_MAP", {})
        or result.config.get("DYNAMIC", {}).get("SECTOR_MAP", {})
        or {}
    )
    if not sector_map:
        sector_map_path = result.config.get("SECTOR_MAP_FILE") or result.config.get(
            "DYNAMIC", {}
        ).get("SECTOR_MAP_FILE")
        sector_map = _load_sector_map(_resolve_config_path(sector_map_path))

    name_map = (
        result.config.get("NAME_MAP", {})
        or result.config.get("DYNAMIC", {}).get("NAME_MAP", {})
        or {}
    )
    if not name_map:
        name_map_path = result.config.get("NAME_MAP_FILE") or result.config.get(
            "DYNAMIC", {}
        ).get("NAME_MAP_FILE")
        if not name_map_path:
            name_map_path = result.config.get("SECTOR_MAP_FILE") or result.config.get(
                "DYNAMIC", {}
            ).get("SECTOR_MAP_FILE")
        name_map = _load_name_map(_resolve_config_path(name_map_path))
        if not name_map:
            project_root = Path(__file__).resolve().parents[2]
            for fallback in (
                project_root / "data" / "processed" / "krx_master_with_sector.csv",
                project_root / "data" / "processed" / "sector_map.csv",
            ):
                name_map = _load_name_map(fallback)
                if name_map:
                    break

    selection_log = list(getattr(result, "dynamic_selection_log", []) or [])
    if selection_log:
        selection_cfg = result.config.get("DYNAMIC", {}).get("SELECTION", {}) or {}
        lookback_years = int(selection_cfg.get("lookback_years", 3))
        (run_dir / "dynamic_selection.json").write_text(
            json.dumps(selection_log, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        rows: list[dict[str, Any]] = []
        seasonal_rows: list[dict[str, Any]] = []
        for entry in selection_log:
            date = entry.get("date")
            season = entry.get("season")
            season_year = entry.get("season_year")
            season_name = entry.get("season_name")
            season_start = entry.get("season_start")
            season_end = entry.get("season_end")
            selected = set(entry.get("selected", []) or [])
            corr_count = entry.get("corr_count", {}) or {}
            momentum = entry.get("momentum", {}) or {}
            corr_rank = entry.get("corr_rank", {}) or {}
            momentum_rank = entry.get("momentum_rank", {}) or {}
            total_score = entry.get("total_score", {}) or {}
            tickers = sorted(set(corr_count) | set(momentum) | selected)
            for t in tickers:
                rows.append(
                    {
                        "date": date,
                        "season": season,
                        "season_year": season_year,
                        "season_name": season_name,
                        "season_start": season_start,
                        "season_end": season_end,
                        "ticker": t,
                        "name": name_map.get(t, ""),
                        "sector": sector_map.get(t, "Unknown"),
                        "selected": t in selected,
                        "corr_count": corr_count.get(t),
                        "momentum": momentum.get(t),
                        "corr_rank": corr_rank.get(t),
                        "momentum_rank": momentum_rank.get(t),
                        "total_score": total_score.get(t),
                    }
                )

            if selected:
                ref_years = _format_ref_years(season_year, lookback_years)
                score_df = pd.DataFrame(
                    {
                        "corr_count": pd.Series(corr_count, dtype="float64"),
                        "momentum": pd.Series(momentum, dtype="float64"),
                        "total_score": pd.Series(total_score, dtype="float64"),
                    }
                )
                if not score_df.empty:
                    if "total_score" not in score_df.columns:
                        score_df["total_score"] = np.nan
                    if score_df["total_score"].isna().all():
                        score_df["total_score"] = score_df["momentum"].fillna(
                            0
                        ) + score_df["corr_count"].fillna(0)
                    score_df = (
                        score_df.reset_index()
                        .rename(columns={"index": "ticker"})
                        .sort_values(
                            ["total_score", "corr_count", "momentum", "ticker"],
                            ascending=[False, False, False, True],
                        )
                    )
                    score_df["rank"] = range(1, len(score_df) + 1)
                    score_df = score_df.set_index("ticker")
                    top_n = int(selection_cfg.get("top_n", 0))
                    if top_n <= 0:
                        top_n = len(score_df)
                    picks = score_df.sort_values("rank").head(top_n)
                    selected_set = set(selected)
                    for t, row in picks.iterrows():
                        seasonal_rows.append(
                            {
                                "date": date,
                                "season": season_name,
                                "season_end": season_end,
                                "ref_years": ref_years,
                                "ticker": t,
                                "name": name_map.get(t, ""),
                                "sector": sector_map.get(t, "Unknown"),
                                "selected": t in selected_set,
                                "rank": (
                                    int(row["rank"]) if pd.notna(row["rank"]) else None
                                ),
                                "total_score": total_score.get(t),
                            }
                        )

        if rows:
            df = pd.DataFrame(
                rows,
                columns=[
                    "date",
                    "season_year",
                    "season_name",
                    "season_start",
                    "season_end",
                    "ticker",
                    "name",
                    "sector",
                    "selected",
                    "corr_count",
                    "momentum",
                    "corr_rank",
                    "momentum_rank",
                    "total_score",
                ],
            )
            if not df.empty:
                num_cols = df.select_dtypes(include=["number"]).columns
                round_cols = [c for c in num_cols if c != "total_score"]
                if round_cols:
                    df.loc[:, round_cols] = df.loc[:, round_cols].round(3)
            df.to_csv(
                run_dir / "dynamic_selection.csv", index=False, encoding="utf-8-sig"
            )

        seasonal_df = pd.DataFrame(
            seasonal_rows,
            columns=[
                "date",
                "season",
                "season_end",
                "ref_years",
                "ticker",
                "name",
                "sector",
                "selected",
                "rank",
                "total_score",
            ],
        )
        if not seasonal_df.empty:
            seasonal_df = seasonal_df.drop_duplicates(
                subset=["date", "ticker", "season_end", "ref_years"], keep="first"
            )
            if "season_end" in seasonal_df.columns and "rank" in seasonal_df.columns:
                seasonal_df = seasonal_df.sort_values(
                    ["date", "season_end", "rank"], ascending=[True, True, True]
                )
        if not seasonal_df.empty:
            num_cols = seasonal_df.select_dtypes(include=["number"]).columns
            round_cols = [c for c in num_cols if c != "total_score"]
            if round_cols:
                seasonal_df.loc[:, round_cols] = seasonal_df.loc[:, round_cols].round(3)
        seasonal_df.to_csv(
            run_dir / "dynamic_selection_seasonal.csv",
            index=False,
            encoding="utf-8-sig",
        )

        sector_rows: list[dict[str, Any]] = []
        for entry in selection_log:
            season_name = entry.get("season_name")
            selected = entry.get("selected", []) or []
            total_score = entry.get("total_score", {}) or {}
            for t in selected:
                sector_rows.append(
                    {
                        "season": season_name,
                        "ticker": t,
                        "sector": sector_map.get(t, "Unknown"),
                        "total_score": total_score.get(t),
                    }
                )
        if sector_rows:
            sector_df = pd.DataFrame(sector_rows)
            sector_summary = (
                sector_df.groupby(["season", "sector"])
                .agg(pick_count=("ticker", "count"), avg_score=("total_score", "mean"))
                .reset_index()
            )
            _round_numeric_df(sector_summary).to_csv(
                run_dir / "sector_rotation.csv",
                index=False,
                encoding="utf-8-sig",
            )
            _ = sector_summary.pivot(
                index="season", columns="sector", values="pick_count"
            ).fillna(0)

    chart_data = None
    if result.equity is not None and not result.equity.empty:
        chart_data = result.equity.to_frame("equity")
        if result.static_equity is not None and not result.static_equity.empty:
            chart_data = chart_data.join(
                result.static_equity.rename("static_equity"), how="outer"
            )
        if result.dynamic_equity is not None and not result.dynamic_equity.empty:
            chart_data = chart_data.join(
                result.dynamic_equity.rename("dynamic_equity"), how="outer"
            )
        if result.benchmark_equity is not None and not result.benchmark_equity.empty:
            chart_data = chart_data.join(
                result.benchmark_equity.rename("benchmark_equity"), how="outer"
            )
        chart_data["drawdown"] = _drawdown_series(chart_data["equity"])
        if "static_equity" in chart_data.columns:
            chart_data["static_drawdown"] = _drawdown_series(
                chart_data["static_equity"]
            )
        if "dynamic_equity" in chart_data.columns:
            chart_data["dynamic_drawdown"] = _drawdown_series(
                chart_data["dynamic_equity"]
            )
        if "benchmark_equity" in chart_data.columns:
            chart_data["benchmark_drawdown"] = _drawdown_series(
                chart_data["benchmark_equity"]
            )

    base_series = (
        result.benchmark_equity
        if result.benchmark_equity is not None and not result.benchmark_equity.empty
        else result.equity
    )
    monthly_heatmap = _build_monthly_heatmap(base_series, years=10)
    _plot_heatmap(
        monthly_heatmap,
        "Seasonality Heatmap (Monthly Return)",
        run_dir / "seasonality_heatmap_monthly.png",
    )

    seasons_cfg = result.config.get("DYNAMIC", {}).get("SEASONS", []) or []
    # If selection mode is auto and selection_log exists, prefer actual selected tickers
    selection_cfg = result.config.get("DYNAMIC", {}).get("SELECTION", {}) or {}
    selection_mode = str(selection_cfg.get("mode", "auto")).lower()
    use_actual_selected = selection_mode == "auto" and bool(selection_log)
    # build a map of latest selection entries by season name
    latest_selection_by_season: dict[str, dict] = {}
    if use_actual_selected:
        try:
            for entry in selection_log:
                sname = entry.get("season_name") or entry.get("season")
                if not sname:
                    continue
                prev = latest_selection_by_season.get(sname)
                # compare by season_year then date
                if prev is None:
                    latest_selection_by_season[sname] = entry
                    continue
                # prefer larger season_year, if equal prefer later date
                try:
                    prev_year = int(prev.get("season_year", -1))
                    prev_date = prev.get("date", "")
                    cur_year = int(entry.get("season_year", -1))
                    cur_date = entry.get("date", "")
                except Exception:
                    latest_selection_by_season[sname] = entry
                    continue
                if cur_year > prev_year or (
                    cur_year == prev_year and cur_date > prev_date
                ):
                    latest_selection_by_season[sname] = entry
        except Exception:
            latest_selection_by_season = {}
    seasonal_heatmap = _build_season_heatmap(base_series, seasons_cfg, years=10)
    _plot_heatmap(
        seasonal_heatmap,
        "Seasonality Heatmap (Season Return)",
        run_dir / "seasonality_heatmap_seasonal.png",
    )

    _plot_cash_weights(
        result.cash_weight,
        result.dynamic_cash_weight,
        result.static_cash_weight,
        run_dir / "cash_weight.png",
    )

    if chart_data is not None and "drawdown" in chart_data.columns:
        _plot_underwater(chart_data["drawdown"], run_dir / "underwater.png")

    _plot_zoom(
        result.equity,
        result.benchmark_equity,
        result.cash_weight,
        "2020-02-15",
        "2020-04-30",
        "Zoom-in: 2020 Mar (COVID)",
        run_dir / "zoom_2020.png",
    )
    _plot_zoom(
        result.equity,
        result.benchmark_equity,
        result.cash_weight,
        "2022-01-01",
        "2022-12-31",
        "Zoom-in: 2022 Bear Market",
        run_dir / "zoom_2022.png",
    )

    _plot_architecture(run_dir / "architecture.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    if result.equity is not None and not result.equity.empty:
        ax.plot(result.equity.index, result.equity.values, label="Backtester")
    if result.benchmark_equity is not None and not result.benchmark_equity.empty:
        ax.plot(
            result.benchmark_equity.index,
            result.benchmark_equity.values,
            label="Benchmark",
        )
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    perf = result.performance or {}
    bench = result.benchmark_performance or {}
    trading_days = int(result.config.get("TRADING_DAYS", 252))
    static_perf = _perf_dict(result.static_equity, result.static_returns, trading_days)
    dynamic_perf = _perf_dict(
        result.dynamic_equity, result.dynamic_returns, trading_days
    )
    static_perf = _format_metrics(static_perf)
    dynamic_perf = _format_metrics(dynamic_perf)
    perf = _format_metrics(perf)
    bench = _format_metrics(bench)
    summary = (
        "Backtester\n"
        f"CAGR: {perf.get('CAGR', float('nan')):.4f}\n"
        f"MDD: {perf.get('MDD', float('nan')):.4f}\n"
        f"Sharpe: {perf.get('Sharpe', float('nan')):.4f}\n"
        f"TotalReturn: {perf.get('TotalReturn', float('nan')):.4f}\n"
        "\nBenchmark\n"
        f"CAGR: {bench.get('CAGR', float('nan')):.4f}\n"
        f"MDD: {bench.get('MDD', float('nan')):.4f}\n"
        f"Sharpe: {bench.get('Sharpe', float('nan')):.4f}\n"
        f"TotalReturn: {bench.get('TotalReturn', float('nan')):.4f}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(run_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    if chart_data is not None and "drawdown" in chart_data.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(chart_data.index, chart_data["drawdown"], label="Backtester")
        if "benchmark_drawdown" in chart_data.columns:
            ax.plot(
                chart_data.index, chart_data["benchmark_drawdown"], label="Benchmark"
            )
        mdd_text = (
            f"MDD (Backtester): {perf.get('MDD', 0.0):.4f}\n"
            f"MDD (Bench): {bench.get('MDD', 0.0):.4f}"
        )
        ax.text(
            0.02,
            0.98,
            mdd_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        ax.set_title("Drawdown")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "drawdown.png", dpi=150)
        plt.close(fig)

    regime_df = _regime_analysis(
        (
            result.benchmark_returns
            if result.benchmark_returns is not None
            and not result.benchmark_returns.empty
            else result.returns
        ),
        result.returns,
    )
    if not regime_df.empty:
        _round_numeric_df(regime_df).to_csv(
            run_dir / "regime_analysis.csv",
            index=False,
            encoding="utf-8-sig",
        )

    seasons = result.config.get("DYNAMIC", {}).get("SEASONS", []) or []
    prices = result.prices
    season_sections: list[str] = []
    for season in seasons:
        name = str(season.get("name", "Season"))
        start_md = str(season.get("start_md", "01-01"))
        end_md = str(season.get("end_md", "12-31"))
        # If in auto mode and we have a latest selection for this season, show that instead
        if use_actual_selected and name in latest_selection_by_season:
            try:
                tickers = list(
                    latest_selection_by_season[name].get("selected", []) or []
                )
            except Exception:
                tickers = [str(t) for t in (season.get("tickers", []) or [])]
        else:
            tickers = [str(t) for t in (season.get("tickers", []) or [])]
        available = [t for t in tickers if t in prices.columns]

        season_sections.append(f"### {name}")
        season_sections.append("")
        season_sections.append(f"- Range: {start_md} ~ {end_md}")
        season_sections.append("")
        season_sections.append("**Tickers**")
        season_sections.append("")
        season_sections.append(", ".join(tickers) if tickers else "(none)")
        season_sections.append("")

        if not available:
            season_sections.append("- No available tickers in price data.")
            season_sections.append("")
            continue

        sub = prices[available].dropna(how="all")
        if sub.empty:
            season_sections.append("- No price data available.")
            season_sections.append("")
            continue

        mask = _season_mask(sub.index, start_md, end_md)
        sub = sub.loc[mask]
        if sub.empty:
            season_sections.append("- No data in season window.")
            season_sections.append("")
            continue

        norm = sub / sub.iloc[0]
        basket = norm.mean(axis=1)
        basket_returns = basket.pct_change(fill_method=None).fillna(0.0)
        basket_perf = _format_metrics(
            performance_summary(basket, basket_returns, trading_days)
        )

        season_sections.append("**Season Basket (equal-weight)**")
        season_sections.append("")
        season_sections.append("| Metric | Value |")
        season_sections.append("|---|---:|")
        season_sections.append(f"| CAGR | {basket_perf['CAGR']:.6f} |")
        season_sections.append(f"| MDD | {basket_perf['MDD']:.6f} |")
        season_sections.append(f"| Sharpe | {basket_perf['Sharpe']:.6f} |")
        season_sections.append(f"| TotalReturn | {basket_perf['TotalReturn']:.6f} |")
        season_sections.append("")

        season_sections.append("**Per-Ticker Return/MDD**")
        season_sections.append("")
        season_sections.append("| Ticker | TotalReturn | MDD |")
        season_sections.append("|---|---:|---:|")
        for t in available:
            s = sub[t].dropna()
            if s.empty:
                continue
            total_ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
            mdd = _mdd_from_equity(s)
            season_sections.append(f"| {t} | {total_ret:.6f} | {mdd:.6f} |")
        season_sections.append("")

    selection_lines: list[str] = []
    if selection_log:
        selection_lines = [
            "## Dynamic Selection",
            "",
            "- Selection log: dynamic_selection.json",
            "- Flat scores: dynamic_selection.csv",
            "- Seasonal top picks: dynamic_selection_seasonal.csv",
            "- Sector rotation: sector_rotation.csv",
            "",
        ]

    mdd_lines: list[str] = []
    try:
        # Single-run MDD defense: replay equity without stop-loss (no second backtest run)
        equity_no_sl = _replay_total_equity_without_stoploss(result)
        mdd_with = _mdd_from_equity(result.equity)
        mdd_without = _mdd_from_equity(equity_no_sl)
        mdd_df = pd.DataFrame(
            [
                {"scenario": "with_stop_loss", "mdd": mdd_with},
                {"scenario": "without_stop_loss", "mdd": mdd_without},
            ]
        )
        _round_numeric_df(mdd_df).to_csv(
            run_dir / "mdd_defense.csv",
            index=False,
            encoding="utf-8-sig",
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["With Stop Loss", "Without Stop Loss"], [mdd_with, mdd_without])
        ax.set_title("MDD Defense Simulation")
        ax.set_ylabel("MDD")
        fig.tight_layout()
        fig.savefig(run_dir / "mdd_defense.png", dpi=150)
        plt.close(fig)
        mdd_lines = [
            "## MDD Defense",
            "",
            "- MDD comparison: mdd_defense.csv",
            "- MDD bar chart: mdd_defense.png",
            "",
        ]
    except Exception:
        mdd_lines = []

    if not regime_df.empty:
        regime_lines = [
            "## Regime Analysis",
            "",
            "- Regime summary: regime_analysis.csv",
            "",
        ]
    else:
        regime_lines = []

    report_seconds = time.perf_counter() - report_start
    optimization_log = {
        "run_id": run_id,
        "report_generation_seconds": round(report_seconds, 4),
        "baseline_seconds": None,
        "optimized_seconds": None,
        "note": "Baseline/optimized timing not captured; add pipeline timing if needed.",
    }
    (run_dir / "optimization_log.json").write_text(
        json.dumps(optimization_log, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    md_lines = [
        f"# Backtester ({config_name})",
        "",
        f"- Run ID: {run_id}",
        "",
        "## Summary",
        "",
        "| Metric | Backtester | Benchmark |",
        "|---|---:|---:|",
        f"| CAGR | {perf.get('CAGR', float('nan')):.6f} | {bench.get('CAGR', float('nan')):.6f} |",
        f"| MDD | {perf.get('MDD', float('nan')):.6f} | {bench.get('MDD', float('nan')):.6f} |",
        f"| Sharpe | {perf.get('Sharpe', float('nan')):.6f} | {bench.get('Sharpe', float('nan')):.6f} |",
        f"| TotalReturn | {perf.get('TotalReturn', float('nan')):.6f} | {bench.get('TotalReturn', float('nan')):.6f} |",
        "",
        "## Strategy Breakdown",
        "",
        "| Metric | Static | Dynamic |",
        "|---|---:|---:|",
        f"| CAGR | {static_perf['CAGR']:.6f} | {dynamic_perf['CAGR']:.6f} |",
        f"| MDD | {static_perf['MDD']:.6f} | {dynamic_perf['MDD']:.6f} |",
        f"| Sharpe | {static_perf['Sharpe']:.6f} | {dynamic_perf['Sharpe']:.6f} |",
        f"| TotalReturn | {static_perf['TotalReturn']:.6f} | {dynamic_perf['TotalReturn']:.6f} |",
        "",
        *selection_lines,
        "## Evidence Outputs",
        "",
        "- Seasonality heatmap (monthly): seasonality_heatmap_monthly.png",
        "- Seasonality heatmap (seasonal): seasonality_heatmap_seasonal.png",
        "- Cash weight dynamics: cash_weight.png",
        "- Underwater plot: underwater.png",
        "- Zoom-in (2020): zoom_2020.png",
        "- Zoom-in (2022): zoom_2022.png",
        "- Architecture diagram: architecture.png",
        "- Optimization log: optimization_log.json",
        *mdd_lines,
        *regime_lines,
        "## Charts",
        "",
        "![Equity Curve](equity_curve.png)",
        "",
        "![Drawdown](drawdown.png)",
        "",
        "## Seasonal Performance",
        "",
        *season_sections,
        "",
        "## Data",
        "",
        "- dynamic_selection.csv",
        "- dynamic_selection_seasonal.csv",
        "- sector_rotation.csv",
        "- regime_analysis.csv",
        "- mdd_defense.csv",
    ]
    (run_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Convenience copies (Direction B): keep root artifacts and also place categorized copies
    _copy_artifacts_to_category_folders(run_dir)

    print(f"[Saved] {run_dir}")


if __name__ == "__main__":
    main()
