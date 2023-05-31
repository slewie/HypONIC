from hyponic.optimizers.base_optimizer import BaseOptimizer

import numpy as np
import numexpr as ne


class CS(BaseOptimizer):
    """
    Cuckoo Search (CS) algorithm

    Hyperparameters:
        + pa(float), default=0.25: probability of cuckoo's egg to be discovered
        + alpha(float), default=0.5: step size
        + k(float), default=1: Levy multiplication coefficient

    Example
    ~~~~~~~
    >>> from hyponic.optimizers.swarm_based.CS import CS
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
    >>> cs = CS(epoch=40, population_size=100, verbose=True, early_stopping=4)
    >>> cs.solve(problem_dict)
    >>> print(cs.get_best_score())
    >>> print(cs.get_best_solution())
    """

    def __init__(self, epoch: int = 10, population_size: int = 10, minmax: str = None, verbose: bool = False,
                 pa: float = 0.25, alpha: float = 0.5, k: float = 1,
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
        :param pa: probability of cuckoo's egg to be discovered
        :param alpha: step size
        :param k: Levy multiplication coefficient
        """
        super().__init__(**kwargs)
        self.epoch = epoch
        self.population_size = population_size
        self.minmax = minmax
        self.verbose = verbose
        self.mode = mode
        self.n_workers = n_workers
        self.early_stopping = early_stopping

        self.pa = pa
        self.alpha = alpha
        self.k = k

        self.nests = None
        self.nests_fitness = None
        self.cuckoo_coords = None

    def _before_initialization(self):
        super()._before_initialization()
        if isinstance(self.pa, float) is False and isinstance(self.pa, int) is False:
            raise TypeError('pa should be a float or an integer')
        if isinstance(self.alpha, float) is False and isinstance(self.alpha, int) is False:
            raise TypeError('alpha should be a float or an integer')
        if isinstance(self.k, float) is False and isinstance(self.k, int) is False:
            raise TypeError('k should be a float or an integer')

    def initialize(self, problem_dict):
        super().initialize(problem_dict)

        self.nests = self.coords
        self.nests_fitness = np.array([self.function(self.nests[i]) for i in range(self.population_size)])
        self.cuckoo_coords = np.random.uniform(self.lb, self.ub, self.dimensions)

    def _levy_flight(self, x):
        u = np.random.normal(0, 1, size=self.dimensions)
        v = np.random.normal(0, 1, size=self.dimensions)
        best_coords = self.nests[self._argminmax()(self.nests_fitness)]
        return ne.evaluate('x + k * u / (abs(v) ** (1 / 1.5)) * (best_coords - x)', local_dict={
            'x': x, 'k': self.k, 'u': u, 'v': v, 'best_coords': best_coords
        })

    def evolve(self, current_epoch):
        x_new = self._levy_flight(self.cuckoo_coords)
        self.cuckoo_coords = np.clip(x_new, self.lb, self.ub)

        next_nest = np.random.randint(0, self.population_size)
        new_fitness = self.function(self.cuckoo_coords)
        if new_fitness < self.nests_fitness[next_nest]:
            self.nests[next_nest] = self.cuckoo_coords
            self.nests_fitness[next_nest] = new_fitness
        number_of_discovered_eggs = int(self.pa * self.population_size)
        worst_nests = np.argsort(self.nests_fitness)[-number_of_discovered_eggs:]
        self.nests[worst_nests] = np.random.uniform(self.lb, self.ub, (number_of_discovered_eggs, self.dimensions))
        self.nests_fitness[worst_nests] = np.array([self.function(self.nests[i]) for i in worst_nests])

    def get_best_score(self):
        return self._minmax()(self.nests_fitness)

    def get_best_solution(self):
        return self.nests[self._argminmax()(self.nests_fitness)]

    def get_current_best_score(self):
        return self.get_best_score()

    def get_current_best_solution(self):
        return self.get_best_solution()
