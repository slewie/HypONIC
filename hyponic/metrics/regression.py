"""
This module contains metrics for evaluating regression models.
"""
from hyponic.metrics.decorators import add_metric_aliases, add_metric_info, maximize_metric, minimize_metric

import numpy as np
import numexpr as ne


@add_metric_aliases("mean_absolute_error")
@add_metric_info("Mean Absolute Error (MAE) is the average of the absolute errors.")
@minimize_metric
def mae(y_true: np.array, y_pred: np.array) -> np.ndarray:
    ae = ne.evaluate("sum(abs(y_true - y_pred))")  # for technical reasons we have to separate summation and division
    return ne.evaluate("ae / length", local_dict={"ae": ae, "length": len(y_true)})


@add_metric_aliases("mean_squared_error")
@add_metric_info("Mean Squared Error (MSE) is the average of the squares of the errors.")
@minimize_metric
def mse(y_true: np.array, y_pred: np.array) -> np.ndarray:
    se = ne.evaluate("sum((y_true - y_pred) ** 2)")
    return ne.evaluate("se / length", local_dict={"se": se, "length": len(y_true)})


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
    ss_res = ne.evaluate("sum((y_true - y_pred) ** 2)")
    sum_y_true = ne.evaluate("sum(y_true)")
    mean = ne.evaluate("sum_y_true / length", local_dict={"sum_y_true": sum_y_true, "length": len(y_true)})
    ss_tot = ne.evaluate("sum((y_true - mean) ** 2)")
    return ne.evaluate("1 - ss_res / ss_tot")


@maximize_metric
def adjusted_r2(y_true: np.array, y_pred: np.array) -> np.ndarray:
    if len(y_true.shape) == 1:
        return r2(y_true, y_pred)
    return 1 - (1 - r2(y_true, y_pred)) * (len(y_true) - 1) / (len(y_true) - len(y_true[0]) - 1)


@minimize_metric
def huber_loss(y_true: np.array, y_pred: np.array, delta: float = 1.0) -> np.ndarray:
    error = y_true - y_pred
    return ne.evaluate("sum(where(abs(error) <= delta, 0.5 * error ** 2, delta * (abs(error) - 0.5 * delta)))")
