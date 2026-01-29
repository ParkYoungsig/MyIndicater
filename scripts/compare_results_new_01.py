from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


METRIC_KEYS = ["CAGR", "MDD", "Sharpe", "TotalReturn"]

# 코드에서 기본값으로 바꿀 수 있는 상수
RESULTS_ROOT_DEFAULT: Optional[str] = r"C:\Users\dudals\Downloads\Team_Project\results"  # 예: "../results"
OUTPUT_ROOT_DEFAULT: Optional[str] = r"C:\Users\dudals\Downloads\Team_Project\results"  # 예: "./outputs"


def _resolve_results_root(path_str: Optional[str]) -> Path:
    if path_str:
        p = Path(path_str)
    else:
        p = project_root() / "results"
    if not p.is_absolute():
        p = (project_root() / p).resolve()
    return p


def _resolve_output_root(path_str: Optional[str], default_root: Path) -> Path:
    if path_str:
        p = Path(path_str)
    else:
        p = default_root
    if not p.is_absolute():
        p = (project_root() / p).resolve()
    return p


def _parse_metric_row(line: str) -> Dict[str, float]:
    # Example: | CAGR | 0.123456 | 0.234567 |
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    if len(parts) < 3:
        return {}
    key = parts[0]
    if key not in METRIC_KEYS:
        return {}
    try:
        back = float(parts[1])
    except ValueError:
        back = float("nan")
    try:
        bench = float(parts[2])
    except ValueError:
        bench = float("nan")
    return {f"{key}_Backtester": back, f"{key}_Benchmark": bench}


def _parse_strategy_row(line: str) -> Dict[str, float]:
    # Example: | CAGR | 0.123456 | 0.234567 |
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    if len(parts) < 3:
        return {}
    key = parts[0]
    if key not in METRIC_KEYS:
        return {}
    try:
        static = float(parts[1])
    except ValueError:
        static = float("nan")
    try:
        dynamic = float(parts[2])
    except ValueError:
        dynamic = float("nan")
    return {f"{key}_Static": static, f"{key}_Dynamic": dynamic}


def _parse_summary(summary_path: Path) -> Dict[str, float]:
    data: Dict[str, float] = {}
    if not summary_path.exists():
        return data

    lines = summary_path.read_text(encoding="utf-8").splitlines()
    in_summary = False
    in_strategy = False
    for line in lines:
        if line.strip().startswith("## Summary"):
            in_summary = True
            in_strategy = False
            continue
        if line.strip().startswith("## Strategy Breakdown"):
            in_strategy = True
            in_summary = False
            continue
        if line.strip().startswith("## "):
            in_summary = False
            in_strategy = False

        if in_summary and line.strip().startswith("|"):
            data.update(_parse_metric_row(line))
        if in_strategy and line.strip().startswith("|"):
            data.update(_parse_strategy_row(line))
    return data


def _collect_runs(results_root: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not results_root.exists():
        return rows

    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.md"
        metrics = _parse_summary(summary_path)
        if not metrics:
            continue
        row: Dict[str, float] = {"run_dir": run_dir.name}
        row.update(metrics)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _print_table(rows: List[Dict[str, float]]) -> None:
    if not rows:
        print("No result summaries found.")
        return

    cols = [
        "run_dir",
        "CAGR_Backtester",
        "MDD_Backtester",
        "Sharpe_Backtester",
        "TotalReturn_Backtester",
    ]
    col_widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = " | ".join(c.ljust(col_widths[c]) for c in cols)
    sep = "-+-".join("-" * col_widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        line = " | ".join(str(r.get(c, "")).ljust(col_widths[c]) for c in cols)
        print(line)


def run(
    *,
    results_root: Optional[str] = RESULTS_ROOT_DEFAULT,
    output_root: Optional[str] = OUTPUT_ROOT_DEFAULT,
    output_prefix: str = "summary_all",
) -> None:
    resolved_results = _resolve_results_root(results_root)
    resolved_output = _resolve_output_root(output_root, resolved_results)
    rows = _collect_runs(resolved_results)

    # Sort by CAGR (backtester) desc if present
    rows.sort(key=lambda r: r.get("CAGR_Backtester", float("-inf")), reverse=True)

    output_csv = resolved_output / f"{output_prefix}.csv"
    output_json = resolved_output / f"{output_prefix}.json"
    _write_csv(output_csv, rows)
    _write_json(output_json, rows)
    _print_table(rows)
    print(f"[Saved] {output_csv}")
    print(f"[Saved] {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backtest results")
    parser.add_argument("--results-root", type=str, default=RESULTS_ROOT_DEFAULT)
    parser.add_argument(
        "--output-root",
        type=str,
        default=OUTPUT_ROOT_DEFAULT,
        help="directory to write summary outputs (default: results-root)",
    )
    parser.add_argument("--output-prefix", type=str, default="summary_all")
    args = parser.parse_args()

    run(
        results_root=args.results_root,
        output_root=args.output_root,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
