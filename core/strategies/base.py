from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
