from __future__ import annotations

from typing import Dict

from infra.api.broker.kis import KISClient

from .base import LiveBroker


class KISBroker(LiveBroker):
    def __init__(
        self,
        mock: bool | None = None,
        *,
        initial_cash: float = 10_000_000,
    ) -> None:
        self.client = KISClient(mock=mock, initial_cash=initial_cash)

    def get_price(self, symbol: str) -> float:
        return self.client.get_price(symbol)

    def get_positions(self) -> Dict[str, int]:
        return self.client.get_positions()

    def get_cash(self) -> float:
        return self.client.get_cash()

    def place_order(
        self, symbol: str, qty: int, side: str, order_type: str = "market"
    ) -> dict:
        return self.client.place_order(symbol, qty, side, order_type=order_type)

    def cancel_order(self, order_id: str) -> dict:
        return self.client.cancel_order(order_id)
