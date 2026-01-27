from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class LiveState:
    last_rebalance_date: Optional[str]
    high_water: Dict[str, float]
    last_target_weights: Dict[str, float]
    cash: Optional[float] = None


class LiveStateStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> LiveState:
        if not self.path.exists():
            return LiveState(
                last_rebalance_date=None,
                high_water={},
                last_target_weights={},
                cash=None,
            )
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return LiveState(
                last_rebalance_date=None,
                high_water={},
                last_target_weights={},
                cash=None,
            )
        return LiveState(
            last_rebalance_date=data.get("last_rebalance_date"),
            high_water={
                str(k): float(v) for k, v in (data.get("high_water") or {}).items()
            },
            last_target_weights={
                str(k): float(v)
                for k, v in (data.get("last_target_weights") or {}).items()
            },
            cash=data.get("cash"),
        )

    def save(self, state: LiveState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_rebalance_date": state.last_rebalance_date,
            "high_water": {k: float(v) for k, v in state.high_water.items()},
            "last_target_weights": {
                k: float(v) for k, v in state.last_target_weights.items()
            },
            "cash": state.cash,
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
