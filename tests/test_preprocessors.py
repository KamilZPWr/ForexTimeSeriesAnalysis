import pandas as pd
import pytest

from src.preprocessors import Preprocessor, StandardPreprocessor
import numpy as np


@pytest.mark.parametrize(
    'label_n, expected_result',
    [
        (1, (np.array([[1, 2, 3], [2, 3, 4]]), np.array([[4], [5]]))),
        (2, (np.array([[1, 2, 3]]), np.array([[4, 5]])))
    ]
)
def test_preprocessor_slice_column(label_n, expected_result):
    series = pd.Series([1, 2, 3, 4, 5])
    windows, unseen_windows = Preprocessor.slice_column(series, 3, label_n)
    assert all((windows == expected_result[0]).reshape(-1))
    assert all((unseen_windows == expected_result[1]).reshape(-1))


def test_standard_preprocessor_run():
    data = pd.DataFrame(data={
        'open': [1, 2, 3, 4, 5],
        'close': [10, 20, 30, 40, 50],
        'high': [100, 200, 300, 400, 500],
        'low': [1000, 2000, 3000, 4000, 5000],
    })
    expected_transformed_data = np.array(
        [
            [
                [10, 1, 100, 1000],
                [20, 2, 200, 2000],
                [30, 3, 300, 3000]
            ],
            [
                [20, 2, 200, 2000],
                [30, 3, 300, 3000],
                [40, 4, 400, 4000],
            ]
        ]
    )
    expected_prediction_labels = np.array(
        [True, True]
    )
    expected_gap_values = np.array(
        [
            [40],
            [50]
        ]
    )

    transformed_data, prediction_labels, gap_values = StandardPreprocessor().run(data, 3, 1)

    assert all((transformed_data == expected_transformed_data).reshape(-1))
    assert all((prediction_labels == expected_prediction_labels).reshape(-1))
    assert all((gap_values == expected_gap_values).reshape(-1))
