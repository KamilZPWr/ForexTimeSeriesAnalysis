from unittest.mock import MagicMock

import pandas as pd
import numpy as np
import pytest

from src.strategies import Move

data = pd.DataFrame(data={
    'open': [1, 2, 3, 4, 5],
    'close': [10, 20, 30, 40, 50],
    'high': [100, 200, 300, 400, 500],
    'low': [1000, 2000, 3000, 4000, 5000],
})
MARGIN = 0.0001
ONE_UNIT = 0.00001


@pytest.fixture()
def engine():
    from src.forex_engine import ForexEngine, Instrument, Interval
    return ForexEngine(Instrument.EUR_USD, Interval.H1, margin=MARGIN, prices=data)


def test_sell_by_index(engine):
    expected_result = (-30 - MARGIN, -30 - MARGIN)
    assert engine.sell_by_index(sell_at=0, close_at=3, lots=ONE_UNIT) == expected_result


def test_buy_by_index(engine):
    expected_result = (30 - MARGIN, 30 - MARGIN,)
    assert engine.buy_by_index(buy_at=0, close_at=3, lots=ONE_UNIT) == expected_result


def test_evaluate_classifier(engine):
    preprocessor = MagicMock()
    preprocessor.run = MagicMock(
        return_value=(
            np.array([[[1], [2], [3]], [[3], [2], [1]]]),
            np.array([1, 0]),
            np.array([[4], [0]])
        )
    )

    model = MagicMock()
    model.predict = MagicMock(return_value=[1, 0])

    strategy = MagicMock()
    strategy.make_decision = MagicMock(
        side_effect=[
            (Move.BUY, 3, ONE_UNIT),
            (Move.SELL, 3, ONE_UNIT)
        ]
    )

    buy_by_index_mock = MagicMock(return_value=(1, 10))
    sell_by_index_mock = MagicMock(return_value=(-1, 5))

    engine.buy_by_index = buy_by_index_mock
    engine.sell_by_index = sell_by_index_mock

    results = engine.evaluate_classifier(
        window_size=3,
        prediction_n=1,
        model=model,
        preprocessing=preprocessor,
        strategy=strategy,
    )
    assert results == ([1, -1], [10, 5])
