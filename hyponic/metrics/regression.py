"""
This module contains metrics for evaluating regression models.
"""
from hyponic.metrics.decorators import add_metric_aliases, add_metric_info, maximize_metric, minimize_metric

import numpy as np


@add_metric_aliases("mean_absolute_error")
@add_metric_info("Mean Absolute Error (MAE) is the average of the absolute errors.")
@minimize_metric
def mae(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred))


@add_metric_aliases("mean_squared_error")
@add_metric_info("Mean Squared Error (MSE) is the average of the squares of the errors.")
@minimize_metric
def mse(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.square(y_true - y_pred))


@add_metric_aliases("root_mean_squared_error")
@add_metric_info("Root Mean Squared Error (RMSE) is the square root of the average of the squared errors.")
@minimize_metric
def rmse(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.sqrt(mse(y_true, y_pred))


@add_metric_aliases("root_mean_squared_log_error")
@add_metric_info("Root Mean Squared Log Error (RMSLE) is the log of the square root of"
                 "the average of the squared errors.")
@minimize_metric
def rmsle(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.log(rmse(y_true, y_pred))


@maximize_metric
def r2(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))


@maximize_metric
def adjusted_r2(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return 1 - (1 - r2(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - len(y_true[0]) - 1)


@minimize_metric
def huber_loss(y_true: np.array, y_pred: np.array, delta: float = 1.0) -> np.ndarray:
    error = y_true - y_pred
    return np.where(np.abs(error) <= delta, 0.5 * np.square(error), delta * (np.abs(error) - 0.5 * delta))
