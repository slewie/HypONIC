from collections import defaultdict

import numpy as np

METRICS_DICT = defaultdict()


def minimize_metric(metric):
    metric.minmax = "min"
    METRICS_DICT[metric.__name__] = metric
    return metric


def maximize_metric(metric):
    metric.minmax = "max"
    METRICS_DICT[metric.__name__] = metric
    return metric


class MetricInfo:
    def __init__(self, info):
        self.__info = info

    def __str__(self):
        if self.__info is None:
            return self.__name__ + " provides no information."

        return self.__info

    def __call__(self, *args, **kwargs):
        print(self.__info)
        return self.__info


def add_metric_info(info):
    def decorator(metric):
        metric.info = MetricInfo(info)
        return metric

    return decorator


@add_metric_info("Mean Squared Error (MSE) is the average of the squared error that is used as the loss function.")
@minimize_metric
def mse(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.square(y_true - y_pred))


@minimize_metric
def mae(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred))


@minimize_metric
def rmse(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


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


@minimize_metric
def log_loss(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.log(1 + np.exp(-y_true * y_pred)))


@minimize_metric
def binary_crossentropy(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


@minimize_metric
def categorical_crossentropy(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


@maximize_metric
def accuracy(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.equal(y_true, y_pred))


@maximize_metric
def precision(y_true: np.array, y_pred: np.array) -> np.ndarray:
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    return tp / (tp + fp)


@maximize_metric
def recall(y_true: np.array, y_pred: np.array) -> np.ndarray:
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    return tp / (tp + fn)


@maximize_metric
def f1(y_true: np.array, y_pred: np.array) -> np.ndarray:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)


@maximize_metric
def fbeta(y_true: np.array, y_pred: np.array, beta: float = 1.0) -> np.ndarray:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


@maximize_metric
def confusion_matrix(y_true: np.array, y_pred: np.array) -> np.ndarray:
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    return np.array([[tp, fp], [fn, tn]])
