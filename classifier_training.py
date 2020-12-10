import json
from functools import partial

import numpy as np
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import BatchNormalization, LSTM, Dropout, Dense
from tensorflow.python.keras.models import Sequential
import pandas as pd

from preprocessing import slice_column


def label_up_down(window: np.array, label: float):
    """
    0 - down
    1 - up
    """
    return window[-1] < label


def transform_data(data, windows_size, label_n):
    windows, labels = slice_column(data.close, windows_size, label_n)
    windows_open, _ = slice_column(data.open, windows_size, label_n)
    windows_high, _ = slice_column(data.high, windows_size, label_n)
    windows_low, _ = slice_column(data.low, windows_size, label_n)

    labels = np.array([
        label_up_down(windows[row_n, :], labels[row_n]) for row_n in range(windows.shape[0])
    ])

    transformed_data = np.array([
        np.array((cl, op, hi, lo)) for cl, op, hi, lo in zip(windows, windows_open, windows_high, windows_low)
    ])

    x_train, x_test, y_train, y_test = train_test_split(
        transformed_data,
        labels,
        test_size=0.2,
        random_state=42
    )
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1])
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return x_train, x_test, y_train, y_test


def load_data():
    data = pd.read_csv('EURUSD-2010_01_04-2020_11_27.csv')
    data.time = pd.to_datetime(data.time)
    return data.drop(columns='time').diff().iloc[1:]


def objective(trial, data):

    window_size = trial.suggest_categorical("window_size", [6, 12, 24, 48, 56, 108])

    x_train, x_test, y_train, y_test = transform_data(data, window_size, 1)

    model = Sequential()

    model.add(
        BatchNormalization()
    )
    model.add(
        LSTM(trial.suggest_categorical("units_lstm", [32, 64, 128, 256, 516, 1032]), input_shape=(window_size, 4))
    )
    model.add(
        Dropout(trial.suggest_float("dropout_lstm", 0.2, 0.6))
    )
    model.add(
        Dense(trial.suggest_categorical("units_dense", [4, 8, 16, 32, 64, 128]))
    )
    model.add(
        Dropout(trial.suggest_float("dropout_dense", 0.2, 0.6))
    )
    model.add(
        Dense(1, activation='relu')
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train,
        y_train,
        epochs=1000,
        validation_data=(x_test, y_test),
        batch_size=64,
        callbacks=[TFKerasPruningCallback(trial, "val_acc")],
    )

    score = model.evaluate(x_test, y_test)
    return score[1]


if __name__ == "__main__":
    hourly_data = load_data()

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(2)
    )

    optimization_function = partial(objective, data=hourly_data)
    study.optimize(optimization_function, n_trials=1000, timeout=604800)

    trial = study.best_trial

    results = {
        'trials_n': len(study.trials),
        'trial_value': trial.value,
        'trial_params': trial.params
    }

    print(results)

    with open('best_trial.json', 'w+') as file:
        json.dump(results, file)
