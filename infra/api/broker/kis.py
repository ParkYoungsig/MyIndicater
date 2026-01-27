from __future__ import annotations

import os
import random
from typing import Dict, Optional

from .base import BrokerAPI


class KISClient(BrokerAPI):
    """KIS Open API 연동 기본 구조 (모의 지원)"""

    def __init__(self, mock: Optional[bool] = None) -> None:
        self.mock = mock if mock is not None else os.getenv("KIS_MOCK", "1") == "1"
        self._positions: Dict[str, int] = {}

    def _ensure_configured(self) -> None:
        if self.mock:
            return
        if not os.getenv("KIS_APP_KEY") or not os.getenv("KIS_APP_SECRET"):
            raise RuntimeError("KIS 환경변수가 설정되지 않았습니다.")

    def get_price(self, symbol: str) -> float:
        self._ensure_configured()
        if self.mock:
            return float(random.randint(10000, 100000))
        raise NotImplementedError("실거래 API 연동은 추후 구현")

    def get_positions(self) -> Dict[str, int]:
        self._ensure_configured()
        return dict(self._positions)

    def place_order(
        self, symbol: str, qty: int, side: str, order_type: str = "market"
    ) -> dict:
        self._ensure_configured()
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if self.mock:
            current = self._positions.get(symbol, 0)
            self._positions[symbol] = current + qty if side == "buy" else current - qty
            return {
                "order_id": f"MOCK-{random.randint(1000, 9999)}",
                "status": "filled",
            }
        raise NotImplementedError("실거래 API 연동은 추후 구현")

    def cancel_order(self, order_id: str) -> dict:
        self._ensure_configured()
        if self.mock:
            return {"order_id": order_id, "status": "canceled"}
        raise NotImplementedError("실거래 API 연동은 추후 구현")
