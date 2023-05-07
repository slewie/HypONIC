"""
This module contains metrics for evaluating classification models.
"""
from hyponic.metrics.decorators import add_metric_info, maximize_metric, minimize_metric

import numpy as np


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
def f1_score(y_true: np.array, y_pred: np.array) -> np.ndarray:
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


@minimize_metric
def log_loss(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.log(1 + np.exp(-y_true * y_pred)))


@minimize_metric
def binary_crossentropy(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


@minimize_metric
def categorical_crossentropy(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
