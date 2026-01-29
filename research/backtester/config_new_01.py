from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from infra.config import load_yaml_config, project_root


DEFAULT_CONFIG: Dict[str, Any] = {
    "START_DATE": "2015-01-01",
    "END_DATE": None,
    "INITIAL_CAPITAL": 100000000,
    "FEES": 0.0015,
    "TRADING_DAYS": 252,
    "FFILL": True,
    "STATIC": {
        "FEES": 0.0004,
        "RATIO": 0.5,
        "ASSETS": [],
        "WEIGHTS": [],
    },
    "DYNAMIC": {
        "FEES": 0.0022,
        "RATIO": 0.5,
        "MOMENTUM_WINDOW": 60,
        "CORRELATION_WINDOW": 60,
        "SELECTION": {
            "mode": "season",
            "top_n": 10,
            "lookback_years": 3,
            "corr_threshold": 0.3,
            "corr_drop_pct": 0.2,
            "momentum_window": 60,
            "corr_window": 60,
            "weighting": "equal",
            "rank_weights": [],
            "score_weights": {"corr": 0.4, "momentum": 0.6},
            "exclude_kr_etf": True,
            "FILTERS": {
                "min_age_days": 750,
                "min_market_cap": 50_000_000_000,
                "min_price": 1000,
                "min_turnover": 30_000_000_000,
                "turnover_window": 60,
            },
        },
        "SEASONS": [
            {"name": "Spring", "start_md": "03-01", "end_md": "05-31", "tickers": []},
            {"name": "Summer", "start_md": "06-01", "end_md": "08-31", "tickers": []},
            {"name": "Fall", "start_md": "09-01", "end_md": "11-30", "tickers": []},
            {"name": "Winter", "start_md": "12-01", "end_md": "02-28", "tickers": []},
        ],
        "LOGIC": {
            "correlation_filter": {
                "type": "average",
                "threshold": 0.4,
                "window": 60,
            },
            "momentum_ranker": {
                "type": "simple",
                "window": 60,
            },
            "selector": {
                "type": "rank_range",
                "rank_range": [4, 7],
                "replacement_order": [1, 2, 3, 8, 9, 10],
            },
            "allocator": {
                "type": "slot_weights",
                "slot_weights": [0.3, 0.3, 0.2, 0.2],
            },
            "stop_loss": {
                "type": "trailing_pct",
                "pct": 0.05,
            },
        },
    },
    "BENCHMARK_TICKER": "069500",
    "DATA": {
        "root": "data",
    },
    "RESULTS": {
        "root": "results",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_logic_profile(user_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """If DYNAMIC.LOGIC_PROFILE is provided, load config/logic_{profile}.yaml.

    - If user also provided DYNAMIC.LOGIC, it wins.
    - If the profile file is missing, raise a clear error.
    """
    dynamic = (
        user_cfg.get("DYNAMIC") if isinstance(user_cfg.get("DYNAMIC"), dict) else {}
    )
    profile = (dynamic or {}).get("LOGIC_PROFILE")
    if not profile:
        return user_cfg

    if isinstance((dynamic or {}).get("LOGIC"), dict) and (dynamic or {}).get("LOGIC"):
        return user_cfg

    profile_name = str(profile).strip()
    try:
        logic_cfg = load_yaml_config(f"logic_{profile_name}")
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"LOGIC_PROFILE='{profile_name}' 설정을 찾지 못했습니다. "
            f"config/logic_{profile_name}.yaml 파일을 생성하세요."
        ) from e

    out = deepcopy(user_cfg)
    out.setdefault("DYNAMIC", {})
    out["DYNAMIC"].setdefault("LOGIC", {})
    if not isinstance(out["DYNAMIC"]["LOGIC"], dict):
        out["DYNAMIC"]["LOGIC"] = {}
    if not isinstance(logic_cfg, dict):
        raise ValueError(f"logic_{profile_name}.yaml은 dict 구조여야 합니다.")

    out["DYNAMIC"]["LOGIC"] = _deep_merge(out["DYNAMIC"]["LOGIC"], logic_cfg)
    return out


def _normalize_logic_name(name: str) -> str:
    raw = str(name).strip()
    if raw.startswith("config."):
        raw = raw[len("config.") :]
    if raw.startswith("config/") or raw.startswith("config\\"):
        raw = raw[7:]
    if raw.endswith(".yaml"):
        raw = raw[:-5]
    return raw


def _apply_logic_override(user_cfg: Dict[str, Any], logic_name: str) -> Dict[str, Any]:
    normalized = _normalize_logic_name(logic_name)
    if not normalized:
        return user_cfg

    try:
        logic_cfg = load_yaml_config(normalized)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"LOGIC override='{logic_name}' 설정을 찾지 못했습니다. "
            f"config/{normalized}.yaml 파일을 생성하세요."
        ) from e

    if not isinstance(logic_cfg, dict):
        raise ValueError(f"{normalized}.yaml은 dict 구조여야 합니다.")

    out = deepcopy(user_cfg)
    out.setdefault("DYNAMIC", {})
    out["DYNAMIC"]["LOGIC"] = logic_cfg
    return out


def _apply_logic_override_dict(
    user_cfg: Dict[str, Any], logic_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    out = deepcopy(user_cfg)
    out.setdefault("DYNAMIC", {})
    out["DYNAMIC"]["LOGIC"] = logic_cfg
    return out


def resolve_logic_config_name(
    user_cfg: Dict[str, Any], logic_name: Optional[str]
) -> Optional[str]:
    if logic_name:
        return _normalize_logic_name(logic_name)

    dynamic = (
        user_cfg.get("DYNAMIC") if isinstance(user_cfg.get("DYNAMIC"), dict) else {}
    )
    if isinstance((dynamic or {}).get("LOGIC"), dict) and (dynamic or {}).get("LOGIC"):
        return None

    profile = (dynamic or {}).get("LOGIC_PROFILE")
    if not profile:
        return None
    profile_name = str(profile).strip()
    return f"logic_{profile_name}"


def _resolve_results_root(cfg: Dict[str, Any]) -> Path:
    results_cfg = cfg.get("RESULTS", {}) or {}
    root = results_cfg.get("root") or "results"
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = (project_root() / root_path).resolve()
    return root_path


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON config must be dict: {path}")
    return data


def _load_json_configs_from_run_dir(
    run_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    backtester_path = run_dir / "config_backtester.json"
    logic_path = run_dir / "config_logic.json"
    backtester_cfg = _load_json_file(backtester_path)
    logic_cfg = _load_json_file(logic_path)
    return backtester_cfg, logic_cfg


def load_dual_engine_config(
    name: str = "backtester",
    *,
    logic_name: Optional[str] = None,
    json_run: Optional[str] = None,
) -> Dict[str, Any]:
    """Load config/{name}.yaml and merge onto DEFAULT_CONFIG."""
    user_cfg = load_yaml_config(name)

    json_cfg = user_cfg.get("JSON") if isinstance(user_cfg.get("JSON"), dict) else {}
    json_enabled = bool(json_cfg.get("enabled", False))
    json_run_name = (
        json_run
        or (json_cfg.get("run_dir") if json_enabled else None)
        or (json_cfg.get("folder") if json_enabled else None)
    )
    json_path = json_cfg.get("path") if json_enabled else None

    if json_run or json_path or json_enabled:
        if json_path and Path(str(json_path)).is_absolute():
            run_dir = Path(str(json_path))
        else:
            if not json_run_name:
                raise ValueError(
                    "JSON 모드가 활성화되었지만 run_dir가 지정되지 않았습니다."
                )
            run_dir = _resolve_results_root(user_cfg) / str(json_run_name)
        # Prefer config_merged.json when available to reproduce the exact configuration used in that run.
        merged_path = run_dir / "config_merged.json"
        if merged_path.exists():
            merged_cfg = _load_json_file(merged_path)
            # config_merged.json already contains the fully merged configuration (including embedded LOGIC dict).
            # CLI start/end overrides will be applied later via apply_overrides().
            user_cfg = merged_cfg
        else:
            backtester_cfg, logic_cfg = _load_json_configs_from_run_dir(run_dir)
            user_cfg = _apply_logic_override_dict(backtester_cfg, logic_cfg)
    else:
        if logic_name:
            user_cfg = _apply_logic_override(user_cfg, logic_name)
        else:
            user_cfg = _apply_logic_profile(user_cfg)
    return _deep_merge(DEFAULT_CONFIG, user_cfg)


def apply_overrides(
    cfg: Dict[str, Any], *, start: Optional[str], end: Optional[str]
) -> Dict[str, Any]:
    out = deepcopy(cfg)
    if start:
        out["START_DATE"] = start
    if end:
        out["END_DATE"] = end
    return out
