import numpy as np


def mse(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.square(y_true - y_pred))


def mae(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.sqrt(np.mean(np.square(y_true - y_pred)))