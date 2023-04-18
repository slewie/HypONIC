from mealpy.swarm_based.PSO import OriginalPSO


class Optimizer:
    def __init__(self, model, X, y, metric, hyperparameters: dict, optimizer=OriginalPSO, **kwargs):
        self.model = model
        self.X = X
        self.y = y
        self.metric = metric
        self.hyperparameters = hyperparameters
        self.kwargs = kwargs
        self.optimizer = optimizer
        self.left_bound = [self.hyperparameters[key][0] for key in self.hyperparameters]
        self.right_bound = [self.hyperparameters[key][1] for key in self.hyperparameters]
        self.hyperparameters_optimized = None
        self.metric_optimized = None

    def objective_function(self, hyperparameters) -> float:
        # TODO: use CV for hyperparameters optimization
        hyperparameters_dict = dict(zip(self.hyperparameters.keys(), hyperparameters))
        self.model.set_params(**hyperparameters_dict)
        self.model.fit(self.X, self.y)
        y_pred = self.model.predict(self.X)
        return self.metric(self.y, y_pred)

    def optimize(self):
        optimizer = OriginalPSO(**self.kwargs)
        self.hyperparameters['fit_func'] = self.objective_function
        self.hyperparameters['lb'] = self.left_bound
        self.hyperparameters['ub'] = self.right_bound
        self.hyperparameters['minmax'] = 'min'  # TODO: it should depends on the metric

        self.hyperparameters_optimized, self.metric_optimized = optimizer.solve(self.hyperparameters)

    def get_optimized_model(self):
        if self.hyperparameters_optimized is None:
            raise ValueError('Model not optimized yet')
        hyperparameters_dict = dict(zip(self.hyperparameters.keys(), self.hyperparameters_optimized))
        self.model.set_params(**hyperparameters_dict)
        self.model.fit(self.X, self.y)
        return self.model

    def get_optimized_parameters(self) -> dict:
        if self.hyperparameters_optimized is None:
            raise ValueError('Model not optimized yet')
        return self.hyperparameters_optimized

    def get_optimized_metric(self) -> float:
        if self.metric_optimized is None:
            raise ValueError('Model not optimized yet')
        return self.metric_optimized



