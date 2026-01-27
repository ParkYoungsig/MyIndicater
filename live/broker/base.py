from __future__ import annotations

from typing import Dict


class LiveBroker:
    def get_price(self, symbol: str) -> float:
        raise NotImplementedError

    def get_cash(self) -> float:
        raise NotImplementedError

    def get_positions(self) -> Dict[str, int]:
        raise NotImplementedError

    def place_order(
        self, symbol: str, qty: int, side: str, order_type: str = "market"
    ) -> dict:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError
