from hyponic.optimizers.base_optimizer import BaseOptimizer

import numpy as np
import numexpr as ne


class PSO(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) algorithm

    Hyperparameters:
        + a1(float), default=0.5: acceleration parameter
        + a2(float), default=0.5: acceleration parameter

    Example
    ~~~~~~~
    >>> from hyponic.optimizers.swarm_based.PSO import PSO
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
    >>> a1, a2 = 0.8, 0.4
    >>> pso = PSO(epoch=40, population_size=100, verbose=True, early_stopping=4, a1=a1, a2=a2)
    >>> pso.solve(problem_dict)
    >>> print(pso.get_best_score())
    >>> print(pso.get_best_solution())
    """
    def __init__(self, epoch: int = 10, population_size: int = 10, minmax: str = None, a1: float = 0.5, a2: float = 0.5,
                 verbose: bool = False, mode: str = 'single', n_workers: int = 4, early_stopping: int | None = None,
                 **kwargs):
        """
        :param epoch: number of iterations
        :param population_size: number of individuals in the population
        :param minmax: 'min' or 'max', depending on whether the problem is a minimization or maximization problem
        :param verbose: whether to print the progress, default is False
        :param mode: 'single' or 'multithread', depending on whether to use multithreading or not
        :param n_workers: number of workers to use in multithreading mode
        :param early_stopping: number of epochs to wait before stopping the optimization process. If None, then early
        stopping is not used
        :param a1, a2: acceleration parameter
        """
        super().__init__(**kwargs)
        self.epoch = epoch
        self.population_size = population_size
        self.minmax = minmax
        self.verbose = verbose
        self.mode = mode
        self.n_workers = n_workers
        self.early_stopping = early_stopping

        self.a1 = a1
        self.a2 = a2
        self.velocities = None
        self.p_best = None
        self.p_best_coords = None
        self.g_best = None
        self.g_best_coords = None

    def _before_initialization(self):
        super()._before_initialization()
        if isinstance(self.a1, float) is False and isinstance(self.a1, int) is False:
            raise ValueError("a1 should be a float or an integer")
        if isinstance(self.a2, float) is False and isinstance(self.a2, int) is False:
            raise ValueError("a2 should be a float or an integer")

    def initialize(self, problem_dict):
        # TODO: if lb and ub are not provided, use the default values
        super().initialize(problem_dict)
        self.g_best = np.inf if self.minmax == "min" else -np.inf
        max_velocity = ne.evaluate("ub - lb", local_dict={'ub': self.ub, 'lb': self.lb})

        self.velocities = np.random.uniform(-max_velocity, max_velocity, size=(self.population_size, self.dimensions))
        self.p_best_coords = self.coords
        self.p_best = np.array([self.function(self.coords[i]) for i in range(self.population_size)])
        self._update_global_best()

    def evolve(self, epoch):
        self.velocities = self._update_velocity()
        self.coords = ne.evaluate("coords + velocities",
                                  local_dict={'coords': self.coords, 'velocities': self.velocities})

        # TODO: if lb or ub is provided, clip the coordinates
        self.coords = np.clip(self.coords, self.lb, self.ub)
        fitness = np.array([self.function(self.coords[i]) for i in range(self.population_size)])
        condition = all(self._minmax()(np.concatenate([self.p_best, fitness])) != self.p_best)

        self.p_best_coords = np.where(condition, self.coords, self.p_best_coords)

        self.p_best = ne.evaluate("where(condition, fitness, p_best)", local_dict={'condition': condition,
                                                                                   'fitness': fitness,
                                                                                   'p_best': self.p_best})
        self._update_global_best()

    def _update_velocity(self):
        r1 = np.random.random()
        r2 = np.random.random()
        expr = "velocities + a1 * r1 * (p_best_coords - coords) + a2 * r2 * (g_best_coords - coords)"
        return ne.evaluate(expr,
                           local_dict={'velocities': self.velocities, 'a1': self.a1, 'a2': self.a2, 'r1': r1, 'r2': r2,
                                       'p_best_coords': self.p_best_coords, 'coords': self.coords,
                                       'g_best_coords': self.g_best_coords})

    def _update_global_best(self):
        if self._minmax()(np.concatenate([self.p_best, [self.g_best]])) != self.g_best:
            self.g_best = self._minmax()(self.p_best)
            self.g_best_coords = self.p_best_coords[self._argminmax()(self.p_best)]

    def get_best_score(self):
        return self.g_best

    def get_best_solution(self):
        return self.g_best_coords

    def get_current_best_score(self):
        return self.p_best

    def get_current_best_solution(self):
        return self.p_best_coords


class IWPSO(PSO):
    """
    Inertia Weight Particle Swarm Optimization

    Hyperparameters:
        + a1(float), default=0.5: acceleration parameter
        + a2(float), default=0.5: acceleration parameter
        + w(float), default=0.5: inertia weight

    Example
    ~~~~~~~
    >>> from hyponic.optimizers.swarm_based.PSO import IWPSO
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
    >>> a1, a2 = 0.8, 0.4
    >>> w = 0.3
    >>> iwpso = IWPSO(epoch=40, population_size=100, verbose=True, early_stopping=4, a1=a1, a2=a2, w=w)
    >>> iwpso.solve(problem_dict)
    >>> print(iwpso.get_best_score())
    >>> print(iwpso.get_best_solution())
    """

    def __init__(self, epoch: int = 10, population_size: int = 10, minmax: str = None, a1: float = 0.5, a2: float = 0.5,
                 w: float = 0.8,
                 verbose: bool = False,
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
        :param a1, a2: acceleration parameter
        :param w: inertia
        """
        super().__init__(epoch, population_size, minmax, a1, a2, verbose, mode, n_workers, early_stopping, **kwargs)
        self.w = w

    def _before_initialization(self):
        super()._before_initialization()
        if isinstance(self.w, float) is False and isinstance(self.w, int) is False:
            raise ValueError("w should be a float or an integer")

    def _update_velocity(self):
        r1 = np.random.random()
        r2 = np.random.random()
        expr = "w * velocities + a1 * r1 * (p_best_coords - coords) + a2 * r2 * (g_best_coords - coords)"
        return ne.evaluate(expr, local_dict={'w': self.w, 'velocities': self.velocities, 'a1': self.a1, 'a2': self.a2,
                                             'r1': r1, 'r2': r2, 'p_best_coords': self.p_best_coords,
                                             'coords': self.coords, 'g_best_coords': self.g_best_coords})
