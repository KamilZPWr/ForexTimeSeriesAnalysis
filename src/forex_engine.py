from datetime import datetime
from enum import Enum, auto
from typing import Tuple, Union, List

import pandas as pd
from tensorflow.python.keras.models import Sequential

from src.preprocessors import ClassifierPreprocessor
from src.strategies import ClassifierStrategy, Move

LOT = 100000


class Interval(Enum):
    H1 = auto()


class Instrument(Enum):
    EUR_USD = auto()


class ForexEngine:
    def __init__(self, instrument: Instrument, interval: Interval, margin: float, prices: pd.DataFrame):

        self.instrument = instrument
        self.interval = interval
        self.margin = margin
        self.data = prices

    def sell(self, sell_at: Union[datetime, int], close_at: Union[datetime, int], lots: float):
        if isinstance(close_at, datetime) and isinstance(sell_at, datetime):
            self.sell_by_datetime(sell_at, close_at, lots)

        if isinstance(close_at, int) and isinstance(sell_at, int):
            self.sell_by_index(sell_at, close_at, lots)

        raise ValueError('Wrong range type, datetimes or ints are expected')

    def sell_by_datetime(self, sell_at: datetime, close_at: datetime, lots: float):
        try:
            sell_at = self.data.datetime.iloc[sell_at]
            close_at = self.data.datetime.iloc[close_at]
        except KeyError:
            raise KeyError('Datetimes out of range')

        return self.sell_by_index(sell_at, close_at, lots)

    def sell_by_index(self, sell_at: int, close_at: int, lots: float) -> Tuple[float, float]:
        sell_price = self.data.iloc[sell_at].close
        close_price = self.data.iloc[close_at].close

        price_change = sell_price - close_price - self.margin
        profit = round(price_change * lots * LOT, 4)
        return price_change, profit

    def buy(self, buy_at: Union[datetime, int], close_at: Union[datetime, int], lots: float):
        if isinstance(close_at, datetime) and isinstance(buy_at, datetime):
            self.buy_by_datetime(buy_at, close_at, lots)

        if isinstance(close_at, int) and isinstance(buy_at, int):
            self.buy_by_index(buy_at, close_at, lots)

        raise ValueError('Wrong range type, datetimes or ints are expected')

    def buy_by_datetime(self, buy_at: datetime, close_at: datetime, lots: float):
        try:
            buy_at = self.data.datetime.iloc[buy_at]
            close_at = self.data.datetime.iloc[close_at]
        except KeyError:
            raise KeyError('Datetimes out of range')

        return self.sell_by_index(buy_at, close_at, lots)

    def buy_by_index(self, buy_at: int, close_at: int, lots: float) -> Tuple[float, float]:
        buy_price = self.data.iloc[buy_at].close
        close_price = self.data.iloc[close_at].close

        price_change = close_price - buy_price - self.margin
        profit = round(price_change * lots * LOT, 4)
        return price_change, profit

    def evaluate_classifier(self, window_size: int, prediction_n: int, model: Sequential,
                            preprocessing: ClassifierPreprocessor,
                            strategy: ClassifierStrategy) -> Tuple[List[float], List[float]]:

        data, labels, gap_values = preprocessing.run(self.data, window_size, prediction_n)
        predictions = model.predict(data)

        iterator = zip(labels, predictions)

        price_changes = []
        profits = []

        for step_n, (label, prediction) in enumerate(iterator):
            decision, close_step, lots = strategy.make_decision(label, prediction, data[step_n, :, 0][-1], gap_values)

            if decision == Move.PASS:
                continue

            current_index = step_n + window_size

            if decision == Move.BUY:
                price_change, profit = self.buy_by_index(current_index, close_step, lots)

            else:
                price_change, profit = self.sell_by_index(current_index, close_step, lots)

            price_changes.append(price_change)
            profits.append(profit)

        return price_changes, profits
