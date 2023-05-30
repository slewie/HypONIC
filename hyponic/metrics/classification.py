"""
This module contains metrics for evaluating classification models.
"""
from hyponic.metrics.decorators import add_metric_info, maximize_metric, minimize_metric

import numpy as np
import numba as nb


@maximize_metric
@add_metric_info("Accuracy is the fraction of predictions model got right.")
@nb.njit
def accuracy(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.equal(y_true, y_pred))


@maximize_metric
@add_metric_info("Precision is the fraction of positive predictions that are correct.")
@nb.njit
def precision(y_true: np.array, y_pred: np.array) -> np.ndarray | float:
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


@maximize_metric
@add_metric_info("Recall is the fraction of positive predictions that are correct")
@nb.njit
def recall(y_true: np.array, y_pred: np.array) -> np.ndarray | float:
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


@maximize_metric
@add_metric_info("F1 score is the harmonic mean of precision and recall.")
@nb.njit
def f1_score(y_true: np.array, y_pred: np.array) -> np.ndarray | float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


@maximize_metric
@add_metric_info("F-beta score is the weighted harmonic mean of precision and recall.")
@nb.njit
def fbeta(y_true: np.array, y_pred: np.array, beta: float = 1.0) -> np.ndarray | float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


@maximize_metric
@add_metric_info(
    "Matthews correlation coefficient is a correlation coefficient between the observed and predicted binary classifications.")
@nb.njit
def confusion_matrix(y_true: np.array, y_pred: np.array) -> np.ndarray:
    tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    return np.array([[tp, fp], [fn, tn]])


@minimize_metric
@add_metric_info(
    "Log loss is the negative log-likelihood of the true labels given a probabilistic classifier’s predictions.")
@nb.njit
def log_loss(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return np.mean(np.log(1 + np.exp(-y_true * y_pred)))


@minimize_metric
@add_metric_info(
    "Binary crossentropy is the negative log-likelihood of the true labels given a probabilistic classifier’s predictions.")
# @nb.njit
def binary_crossentropy(y_true: np.array, y_pred: np.array) -> np.ndarray:
    # TODO: fix that logarithm can be negative or zero
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


@minimize_metric
@add_metric_info(
    "Categorical crossentropy is the negative log-likelihood of the true labels given a probabilistic classifier’s predictions.")
@nb.njit
def categorical_crossentropy(y_true: np.array, y_pred: np.array) -> np.ndarray:
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))
