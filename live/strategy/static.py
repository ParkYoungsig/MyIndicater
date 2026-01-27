from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        return {k: 0.0 for k in weights.keys()}
    return {k: float(v) / total for k, v in weights.items()}


def _freq_to_pandas(freq: str) -> str:
    mapping = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "ME",
        "Quarterly": "QE",
    }
    return mapping.get(freq, freq)


@dataclass
class LiveStaticStrategy:
    config: Dict[str, object]

    def __post_init__(self) -> None:
        config = self.config if isinstance(self.config, dict) else {}
        static_cfg = config.get("STATIC", {}) or {}
        if not isinstance(static_cfg, dict):
            static_cfg = {}
        self.assets = static_cfg.get("ASSETS", {}) or {}
        weights = static_cfg.get("WEIGHTS", []) or []
        self.rebalance_freq = str(static_cfg.get("REBALANCE_FREQ", "Monthly"))

        tickers = list(self.assets.values())
        if isinstance(weights, list) and len(weights) == len(tickers):
            self.weights = {t: float(w) for t, w in zip(tickers, weights)}
        else:
            n = len(tickers)
            self.weights = {t: (1.0 / n if n > 0 else 0.0) for t in tickers}

    def rebalance_dates(self, index: pd.Index) -> set[pd.Timestamp]:
        freq = _freq_to_pandas(self.rebalance_freq)
        anchor = pd.Series(1, index=pd.DatetimeIndex(index))
        return set(anchor.resample(freq).last().index)

    def target_weights(self) -> Dict[str, float]:
        return _normalize(self.weights)

    def tickers(self) -> List[str]:
        return list(self.weights.keys())
