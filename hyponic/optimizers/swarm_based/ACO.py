from hyponic.optimizers.base_optimizer import BaseOptimizer

import numpy as np


class ACO(BaseOptimizer):
    """
    Ant Colony Optimization (ACO)

    Hyperparameters:
        + alpha(float), default=1: the importance of pheromone
        + beta(float), default=1: the importance of heuristic information
        + rho(float), default=0.5: the pheromone evaporation rate
        + q(float), default=1: the pheromone intensity

    Example
    ~~~~~~~
    >>> from hyponic.optimizers.swarm_based.ACO import ACO
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
    >>> alpha = 1
    >>> beta = 1
    >>> rho = 0.5
    >>> q = 1
    >>>
    >>> aco = ACO(epoch=40, population_size=100, verbose=True, early_stopping=4, alpha=alpha, beta=beta, rho=rho, q=q)
    >>> aco.solve(problem_dict)
    >>> print(aco.get_best_score())
    >>> print(aco.get_best_solution())
    """

    def __init__(self, epoch: int = 10, population_size: int = 10, minmax: str = None, alpha: float = 1,
                 beta: float = 1, rho: float = 0.5, q: float = 1,
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
        :param alpha: the importance of pheromone
        :param beta: the importance of heuristic information
        :param rho: the pheromone evaporation rate
        :param q: the pheromone intensity
        """
        super().__init__(**kwargs)
        self.epoch = epoch
        self.population_size = population_size
        self.minmax = minmax
        self.verbose = verbose
        self.mode = mode
        self.n_workers = n_workers
        self.early_stopping = early_stopping

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

        self.pheromone = None
        self.population = None
        self.scores = None
        self.best_score = None
        self.best_solution = None

    def _before_initialization(self):
        super()._before_initialization()
        if self.alpha < 0:
            raise ValueError("alpha should be a positive float")
        if self.beta < 0:
            raise ValueError("beta should be a positive float")
        if self.rho < 0 or self.rho > 1:
            raise ValueError("rho should be a float between 0 and 1")
        if self.q < 0:
            raise ValueError("q should be a positive float")

    def initialize(self, problem_dict):
        super().initialize(problem_dict)

        self.pheromone = np.ones((self.population_size, self.dimensions))
        self.best_score = np.inf if self.minmax == 'min' else -np.inf

        self.population = np.random.uniform(low=self.lb, high=self.ub, size=(self.population_size, self.dimensions))
        self.scores = np.zeros(self.population_size)
        for i in range(self.population_size):
            self.scores[i] = self.function(self.population[i])
            if self._minmax()(self.scores[i]) < self._minmax()(self.best_score):
                self.best_score = self.scores[i]
                self.best_solution = self.population[i]

    def evolve(self, epoch):
        new_population = np.zeros((self.population_size, self.dimensions))
        new_scores = np.zeros(self.population_size)

        for i in range(self.population_size):
            ant_position = np.random.uniform(low=self.lb, high=self.ub, size=self.dimensions)

            for j in range(self.dimensions):
                pheromone_weights = self.pheromone[:, j] ** self.alpha
                heuristic_weights = 1 / (np.abs(ant_position[j] - self.population[:, j]) + 1e-6) ** self.beta
                probs = pheromone_weights * heuristic_weights
                probs /= np.sum(probs)
                ant_position[j] = np.random.choice(self.population[:, j], p=probs)

            ant_score = self.function(ant_position)
            self._update_pheromone(ant_score)

            if self._minmax()(ant_score) < self._minmax()(self.best_score):
                self.best_solution = ant_position
                self.best_score = ant_score

            new_population[i] = ant_position
            new_scores[i] = ant_score

        self.population = new_population
        self.scores = new_scores

    def get_best_score(self):
        return self.best_score

    def get_best_solution(self):
        return self.best_solution

    def get_current_best_score(self):
        return self._minmax()(self.scores.min())

    def get_current_best_solution(self):
        return self.population[self.scores.argmin()]

    def _initialize_pheromone(self):
        self.pheromone = np.ones((self.population_size, self.dimensions))

    def _update_pheromone(self, ant_score):
        delta_pheromone = np.zeros((self.population_size, self.dimensions))
        delta_pheromone[:, :] = self.q / ant_score
        self.pheromone = (1 - self.rho) * self.pheromone + self.rho * delta_pheromone
