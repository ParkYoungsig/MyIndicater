from __future__ import annotations

from typing import Dict

from live.broker.base import LiveBroker


class OrderManager:
    def __init__(self, broker: LiveBroker) -> None:
        self.broker = broker

    def rebalance_to_target_positions(
        self, target_positions: Dict[str, int]
    ) -> list[dict]:
        current = self.broker.get_positions()
        orders = []

        for symbol, target_qty in target_positions.items():
            current_qty = current.get(symbol, 0)
            diff = target_qty - current_qty
            if diff == 0:
                continue
            side = "buy" if diff > 0 else "sell"
            order = self.broker.place_order(symbol, abs(diff), side)
            orders.append(order)

        return orders
