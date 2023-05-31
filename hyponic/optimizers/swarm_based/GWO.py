from hyponic.optimizers.base_optimizer import BaseOptimizer

import numpy as np
import numexpr as ne


class GWO(BaseOptimizer):
    """
    Grey Wolf Optimization (GWO) algorithm

    Example
    ~~~~~~~
    >>> from hyponic.optimizers.swarm_based.GWO import GWO
    >>> import numpy as np
    >>>
    >>> def sphere(x):
    >>>     return np.sum(x ** 2)
    >>>
    >>> problem_dict = {
    >>>     'fit_func': sphere,
    >>>     'lb': [-5.12, -5, -14, -6, -0.9],
    >>>     'ub': [5.12, 5, 14, 6, 0.9],
    >>>     'minmax': 'min'
    >>> }
    >>>
    >>> gwo = GWO(epoch=40, population_size=100, verbose=True, early_stopping=4)
    >>> gwo.solve(problem_dict)
    >>> print(gwo.get_best_score())
    >>> print(gwo.get_best_solution())
    """

    def __init__(self, epoch: int = 10, population_size: int = 10, minmax: str = None, verbose: bool = False,
                 mode: str = 'single', n_workers: int = 4, early_stopping: int | None = None, **kwargs):
        """
        :param epoch: number of iterations
        :param population_size: number of individuals in the population
        :param minmax: 'min' or 'max', depending on whether the problem is a minimization or maximization problem
        :param verbose: whether to print the progress, default is False
        :param mode: 'single' or 'multithread', depending on whether to use multithreading or not
        :param n_workers: number of workers to use in multithreading mode
        :param early_stopping: number of epochs to wait before stopping the optimization process. If None, then early
        stopping is not used
        """
        super().__init__(**kwargs)
        self.epoch = epoch
        self.population_size = population_size
        self.minmax = minmax
        self.verbose = verbose
        self.mode = mode
        self.n_workers = n_workers
        self.early_stopping = early_stopping

        self.fitness = None
        self.g_best = None
        self.g_best_coords = None

    def initialize(self, problem_dict):
        super().initialize(problem_dict)

        self.g_best = np.inf if self.minmax == 'min' else -np.inf
        self.g_best_coords = np.zeros(self.dimensions)
        self.fitness = np.array([self.function(x) for x in self.coords], dtype=np.float64)

    def evolve(self, current_epoch):
        a = 2 - current_epoch * (2 / self.epoch)

        for i in range(self.population_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2

            D = np.abs(C * self.coords[np.random.randint(0, self.population_size)] - self.coords[np.random.randint(0, self.population_size)])
            coords_new = self.coords[i] + A * D
            fitness_new = self.function(coords_new)
            if self._minmax()([fitness_new, self.fitness[i]]) == fitness_new:
                self.coords[i] = coords_new
                self.fitness[i] = fitness_new

            if self._minmax()([fitness_new, self.g_best]) == fitness_new:
                self.g_best = fitness_new
                self.g_best_coords = coords_new

    def get_best_score(self):
        return self.g_best

    def get_best_solution(self):
        return self.g_best_coords

    def get_current_best_score(self):
        return self._minmax()(self.fitness)

    def get_current_best_solution(self):
        return self.coords[self._argminmax()(self.fitness)]

