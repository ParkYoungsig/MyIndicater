from __future__ import annotations

from typing import Callable, Dict, Optional

from infra.config import load_yaml_config
from infra.logger import get_logger
from live.broker.kis import KISBroker
from live.execution.order_manager import OrderManager
from live.monitoring.slack import SlackNotifier


WeightFunc = Callable[[Dict[str, float]], Dict[str, float]]


class LiveEngine:
    def __init__(
        self, weight_func: Optional[WeightFunc] = None, mock: bool = True
    ) -> None:
        self.config = load_yaml_config("live")
        self.universe_config = load_yaml_config("universe")
        self.logger = get_logger("live.engine")
        self.broker = KISBroker(mock=mock)
        self.order_manager = OrderManager(self.broker)
        self.notifier = SlackNotifier(self.config.get("slack_webhook"))
        self.weight_func = weight_func or self._default_weight_func

    def _default_weight_func(self, prices: Dict[str, float]) -> Dict[str, float]:
        symbols = list(prices.keys())
        if not symbols:
            return {}
        w = 1.0 / len(symbols)
        return {s: w for s in symbols}

    def _get_universe(self) -> list[str]:
        symbols = self.config.get("universe", [])
        if symbols:
            return symbols
        key = self.config.get("universe_key")
        if key:
            return self.universe_config.get(key, [])
        return []

    def _get_prices(self) -> Dict[str, float]:
        symbols = self._get_universe()
        return {s: self.broker.get_price(s) for s in symbols}

    def _target_positions(self, prices: Dict[str, float]) -> Dict[str, int]:
        target_weights = self.weight_func(prices)
        total_cash = self.config.get("initial_cash", 10_000_000)
        target_positions = {}
        for symbol, weight in target_weights.items():
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            qty = int((total_cash * weight) / price)
            target_positions[symbol] = qty
        return target_positions

    def run_once(self) -> None:
        prices = self._get_prices()
        target_positions = self._target_positions(prices)
        orders = self.order_manager.rebalance_to_target_positions(target_positions)
        if orders:
            self.notifier.notify(f"주문 체결: {orders}")
        self.logger.info("run_once completed")
