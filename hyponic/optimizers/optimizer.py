from mealpy.swarm_based.PSO import OriginalPSO

from typing import Callable

from hyponic.space import Space
from functools import partial


class HypONIC:
    # XXX: Should be moved from optimizers folder
    def __init__(self, model, X, y, metric: Callable, optimizer=None, **kwargs):
        self.model = model
        self.X = X
        self.y = y

        self.metric = metric
        if kwargs is None: # Default values for optimizer
            kwargs = {"epoch": 10, "pop_size": 10}

        if optimizer is None:
            self.optimizer = OriginalPSO(**kwargs)
        else:
            self.optimizer = optimizer(**kwargs)

        self.hyperparams_optimized = None
        self.metric_optimized = None

    @staticmethod
    def not_optimized_error(func):
        """
        Decorator for checking if model is optimized
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

    def optimize(self, hyperparams: dict, verbose=False) -> (dict, float):
        # Create a space for hyperparameters
        hyperspace = Space(hyperparams)
        if verbose:
            print("Successfully created a space for hyperparameters optimization")
            print(hyperspace)

        # Map hyperparameters to continuous space
        mappings_with_bounds = hyperspace.get_continuous_mappings(origins=0)

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
            "minmax": "min"  # TODO: it should depends on the metric
        }

        hyperparams_optimized, metric_optimized = self.optimizer.solve(paramspace)

        # Map back to original space
        hyperparams_optimized = {
            dim: mapping_funcs[dim](val) for dim, val in zip(hyperspace.dimensions_names, hyperparams_optimized)
        }

        self.hyperparams_optimized = hyperparams_optimized
        self.metric_optimized = metric_optimized

        return hyperparams_optimized, metric_optimized

    @not_optimized_error
    def get_optimized_model(self):
        self.model.set_params(**self.hyperparams_optimized)
        self.model.fit(self.X, self.y)
        return self.model

    @not_optimized_error
    def get_optimized_parameters(self) -> dict:
        return self.hyperparams_optimized

    @not_optimized_error
    def get_optimized_metric(self) -> float:
        return self.metric_optimized
