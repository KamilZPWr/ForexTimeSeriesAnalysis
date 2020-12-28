from abc import abstractmethod
from enum import Enum, auto
from typing import Tuple

import numpy as np


class Move(Enum):
    BUY = auto()
    SELL = auto()
    PASS = auto()


class ClassifierStrategy:
    @abstractmethod
    def make_decision(self, label: int, prediction: int, close_price: float,
                      gap_values: np.array) -> Tuple[Move, int, float]:
        """
        Decide what move should be taken for given state.

        :param label: Annotated label
        :param prediction: Predicted label
        :param close_price: Window last close price
        :param gap_values: Close prices between last seen close price and prediction (including)
        :return: Tuple of move to take, step on which move should be closed, lots
        """
        pass
