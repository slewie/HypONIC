"""
This module contains decorators for metrics.

The following decorators are available:
- add_metric_to_dict -- a decorator that adds the metric to the dictionary of metrics
- minimize_metric -- a decorator for metrics that should be minimized
- maximize_metric -- a decorator for metrics that should be maximized

This module also contains a dictionary of metrics to which metrics are added
when they are decorated. This dictionary is used by the HypONIC class to
retrieve the metrics passed to it as the string literals. This is helpful,
because one metric can have multiple aliases, and the user can pass any of
them to the HypONIC class as a string.
"""
from collections import defaultdict
from typing import Callable

"""
A dictionary of aliased metrics to which metrics are added when they are decorated.
"""
METRICS_DICT = defaultdict()


def add_metric_to_dict(metric, aliases: list = None) -> Callable:
    """
    A decorator that adds the metric to the dictionary of metrics.
    :param metric: the metric to be added to the dictionary
    :param aliases: a list of aliases for the metric
    :return: decorated metric
    """
    METRICS_DICT[metric.__name__] = metric

    # Add aliases to the dictionary
    if aliases is not None:
        for alias in aliases:
            METRICS_DICT[alias] = metric

    return metric


def add_metric_info(info):
    """
    A decorator that adds information about the metric.
    :param info: information about the metric
    :return: decorated metric
    """

    def decorator(metric):
        metric.info = PrintableProperty(info)
        return metric

    return decorator


def maximize_metric(aliases: list = None) -> Callable:
    """
    A decorator for metrics that should be maximized.
    :param aliases: a list of aliases for the metric
    :return: decorated metric
    """

    def decorator(metric):
        metric.minmax = "max"
        return add_metric_to_dict(metric, aliases)

    return decorator


def minimize_metric(aliases: list = None) -> Callable:
    """
    A decorator for metrics that should be minimized.
    :param aliases: a list of aliases for the metric
    :return: decorated metric
    """

    def decorator(metric):
        metric.minmax = "min"
        return add_metric_to_dict(metric, aliases)

    return decorator


class PrintableProperty:
    """
    A class for the properties that should either be printed or returned as a string.

    Use case:
    mse.info()  # will print the information about the metric
    mse.info    # will return the information about the metric as a string
    """

    def __init__(self, text):
        self.__text = text

    def __str__(self):
        return self.__text

    def __repr__(self):
        return self.__text

    def __call__(self, *args, **kwargs):
        print(self.__text)
        return self.__text
