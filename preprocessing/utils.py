from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def slice_column(column: pd.Series, window_size: int, label_n: int = 0) -> Union[np.array, Tuple[np.array, np.array]]:
    items_n = column.shape[0]

    last_slice_n = items_n - window_size
    last_slice_n -= label_n

    windows = np.array([
        column[n: n + window_size] for n in range(last_slice_n)
    ])

    if label_n:
        labels = np.array([
            column[window_size + n + label_n] for n in range(last_slice_n)
        ])

        return windows, labels
    return windows


def rescale_window(window: np.array, label: float = None, scaler=None):
    if not scaler:
        scaler = MinMaxScaler().fit(window)

    window = scaler.transform(window)[:, 0]

    if label:
        label = scaler.transform(np.array([label]).reshape(-1, 1))[0][0]
        return window, label
    return window


def rescale_windows_with_labels(windows: np.array,
                                labels: np.array = np.array([])) -> Tuple[np.array, np.array, List[MinMaxScaler]]:
    rescaled_windows = []
    rescaled_labels = []
    scalers = []
    for row_n in range(windows.shape[0]):
        window = windows[row_n].reshape(-1, 1)

        data_fit = np.concatenate((window, labels[row_n]), axis=None)
        data_fit = data_fit.reshape(-1, 1)

        scaler = MinMaxScaler()
        scaler.fit(data_fit)
        scalers.append(scaler)

        window, label = rescale_window(window, labels[row_n], scaler)
        rescaled_windows.append(window)
        rescaled_labels.append([label])

    windows = np.array(rescaled_windows)
    labels = np.array(rescaled_labels)
    return windows, labels, scalers


def rescale_windows(windows: np.array) -> Tuple[np.array, List[MinMaxScaler]]:
    rescaled_windows = []
    scalers = []
    for row_n in range(windows.shape[0]):
        window = windows[row_n].reshape(-1, 1)

        scaler = MinMaxScaler()
        scaler.fit(window)
        scalers.append(scaler)

        window = rescale_window(window, scaler=scaler)
        rescaled_windows.append(window)

    windows = np.array(rescaled_windows)
    return windows, scalers


def unscale_predictions(predictions: List[float], scalers: List[MinMaxScaler]) -> np.array:
    unscaled_values = []
    for n, scaler in enumerate(scalers):
        unscaled_values.append(
            scaler.inverse_transform(np.array([predictions[n]]).reshape(-1, 1))[0]
        )

    return np.array(unscaled_values).reshape(-1)
