from warnings import warn

from hyponic.optimizers.swarm_based.PSO import IWPSO
from hyponic.metrics.decorators import METRICS_DICT
from hyponic.utils.problem_identifier import ProblemIdentifier, ProblemType
from hyponic.config.sklearn_models import models_dict

from typing import Callable

from hyponic.space import Space
from functools import partial
import numpy as np


class HypONIC:
    """
    HypONIC (Hyperparameter Optimization using Nature-Inspired Computing)

    Main class for hyperparameter optimization.
    """

    def __init__(self, model, X, y, metric: Callable | str | None = None, optimizer=None, **kwargs):
        self.model = model
        try:
            self.X = np.array(X)
            self.y = np.ravel(np.array(y))
        except Exception as e:
            raise Exception(f"X and y must be convertible to numpy array. Error: {e}")

        if isinstance(metric, str):
            # Try to get metric from the METRICS_DICT
            self.metric = METRICS_DICT.get(metric, None)

            if self.metric is None:
                raise Exception(f"Metric {metric} is not found.")
        elif isinstance(metric, Callable):
            self.metric = metric
        elif metric is None:
            # If metric is None, then try to get metric from the problem type
            problem_type = ProblemIdentifier(self.y).get_problem_type()

            match problem_type:
                case ProblemType.REGRESSION:
                    self.metric = METRICS_DICT["mse"]
                case ProblemType.BINARY_CLASSIFICATION:
                    self.metric = METRICS_DICT["binary_crossentropy"]
                case ProblemType.MULTICLASS_CLASSIFICATION:
                    self.metric = METRICS_DICT["log_loss"]
        else:
            raise Exception(f"Metric {metric} is not found.")

        try:
            self.minmax = self.metric.__getattribute__("minmax")
        except AttributeError:
            # If a metric does not have minmax attribute,
            # then it is assumed to be a custom metric and will be minimized by default
            warn(f"Metric {metric.__name__} does not have minmax attribute. Minimize by default.")
            self.minmax = "min"

        if kwargs is None:  # Default values for optimizer
            kwargs = {"epoch": 10, "pop_size": 10}

        if optimizer is None:
            self.optimizer = IWPSO(**kwargs)
        else:
            self.optimizer = optimizer(**kwargs)

        self.hyperparams_optimized = None
        self.metric_optimized = None

    @staticmethod
    def warn_not_optimized(func):
        """
        Decorator that warns if a method is called before optimization.
        """

        def wrapper(*args, **kwargs):
            if args[0].hyperparams_optimized is None:
                raise Exception("Model is not optimized yet. Please call optimize method first")
            return func(*args, **kwargs)

        return wrapper

    def _fitness_wrapper(self, dimensions_names: list, mapping_funcs: dict, values: list) -> float:
        # Map back to original space
        hyperparams = {
            dim: mapping_funcs[dim](val) for dim, val in zip(dimensions_names, values)
        }

        self.model.set_params(**hyperparams)
        self.model.fit(self.X, self.y)

        y_pred = self.model.predict(self.X)
        return self.metric(self.y, y_pred)  # TODO: maybe cross-validation could be used instead

    def optimize(self, hyperparams: dict | None = None, verbose=False) -> (dict, float):
        if hyperparams is None:
            hyperparams = models_dict.get(str(self.model.__class__), dict())

        # Create a space for hyperparameters
        hyperspace = Space(hyperparams)
        if verbose:
            print("Successfully created a space for hyperparameters optimization")
            print(f"Using {self.optimizer.__class__.__name__} optimizer")
            print(f"Metric {self.metric.__name__} is subject to {self.minmax}imization")
            print(hyperspace)

        # Map hyperparameters to continuous space
        mappings_with_bounds = hyperspace.get_continuous_mappings(origins=0)  # Make that all dimensions start from 0

        # Split mappings and bounds
        mapping_funcs = {}

        low_bounds = []
        highs_bounds = []

        for name in hyperspace.dimensions_names:
            mapping, (low, high) = mappings_with_bounds[name]

            mapping_funcs[name] = mapping
            low_bounds.append(low)
            highs_bounds.append(high)

        paramspace = {
            "fit_func": partial(self._fitness_wrapper, hyperspace.dimensions_names, mapping_funcs),
            "lb": low_bounds,
            "ub": highs_bounds,
            "minmax": self.minmax
        }

        hyperparams_optimized, metric_optimized = self.optimizer.solve(paramspace, verbose=verbose)

        # Map back to the original space
        hyperparams_optimized = {
            dim: mapping_funcs[dim](val) for dim, val in zip(hyperspace.dimensions_names, hyperparams_optimized)
        }

        self.hyperparams_optimized = hyperparams_optimized
        self.metric_optimized = metric_optimized

        return hyperparams_optimized, metric_optimized

    @warn_not_optimized
    def get_optimized_model(self):
        self.model.set_params(**self.hyperparams_optimized)
        self.model.fit(self.X, self.y)
        return self.model

    @warn_not_optimized
    def get_optimized_parameters(self) -> dict:
        return self.hyperparams_optimized

    @warn_not_optimized
    def get_optimized_metric(self) -> float:
        return self.metric_optimized

    @warn_not_optimized
    def visualize_history_fitness(self):
        self.optimizer.visualize_history_fitness()

    @warn_not_optimized
    def visualize_history_time(self):
        self.optimizer.visualize_history_time()
