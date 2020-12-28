from abc import abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd


class Preprocessor:
    @staticmethod
    def slice_column(column: pd.Series, window_size: int, label_n: int = 1) -> Tuple[np.array, np.array]:
        """
        Create price windows.

        :param column: Column to slice.
        :param window_size: No. of elements in window.
        :param label_n: No. of value to predict.
        :return: Tuple of values to make decision on, labels, values between window and label
        """

        items_n = column.shape[0]
        label_n -= 1
        last_slice_n = items_n - window_size
        last_slice_n -= label_n if label_n != 0 else 0

        windows = np.array([
            column[n: n + window_size] for n in range(last_slice_n)
        ])

        if label_n == 0:
            unseen_windows = np.array([
                column[window_size + n] for n in range(last_slice_n)
            ])
        else:
            unseen_windows = np.array([
                column[window_size + n: window_size + n + label_n+1] for n in range(last_slice_n)
            ])

        return windows, unseen_windows.reshape(last_slice_n, -1)


class ClassifierPreprocessor(Preprocessor):
    @abstractmethod
    def run(self, data: pd.DataFrame, window_size: int,
            label_n: int) -> Tuple[np.array, np.array, np.array]:
        """
        Preprocess data to feed classifier.

        :param data: DataFrame with features and datetimes
        :param window_size: No. of elements to predict on.
        :param label_n: No. of element to predict (counting from last window's item)
                        ex. label_n = 1 will use window_size+1 as label index
        :return: Tuple of transformed data, labels to predict, value under labels, close prices
        """
        pass

    @staticmethod
    def label_up_down(window: np.array, label: float):
        """
        0 - down
        1 - up
        """
        return window[-1] < label


class StandardPreprocessor(ClassifierPreprocessor):
    def run(self, data: pd.DataFrame, windows_size: int, label_n: int) -> Tuple[np.array, np.array, np.array]:

        windows_close, gap_values = self.slice_column(data.close, windows_size, label_n)
        windows_open, _ = self.slice_column(data.open, windows_size, label_n)
        windows_high, _ = self.slice_column(data.high, windows_size, label_n)
        windows_low, _ = self.slice_column(data.low, windows_size, label_n)

        prediction_labels = np.array([
            self.label_up_down(windows_close[row_n, :], gap_values[row_n, -1])
            for row_n in range(windows_close.shape[0])
        ])

        transformed_data = np.array([
            np.array((cl, op, hi, lo)).T
            for cl, op, hi, lo in zip(windows_close, windows_open, windows_high, windows_low)
        ])

        prediction_labels = prediction_labels.reshape(-1)

        return transformed_data, prediction_labels, gap_values
