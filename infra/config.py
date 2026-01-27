from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml_config(name: str) -> Dict[str, Any]:
    """config/{name}.yaml 로드"""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML이 필요합니다. `pip install pyyaml` 실행 후 재시도하세요."
        ) from exc

    config_path = project_root() / "config" / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data
