from hyponic.optimizers.base_optimizer import BaseOptimizer

import numpy as np
import numexpr as ne


class ABC(BaseOptimizer):
    def __init__(self, epoch: int = 10, population_size: int = 10, minmax: str = None, limits=25,
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
        :param limits: the number of trials before abandoning food source
        """
        super().__init__(**kwargs)
        self.epoch = epoch
        self.population_size = population_size
        self.minmax = minmax
        self.verbose = verbose
        self.mode = mode
        self.n_workers = n_workers
        self.early_stopping = early_stopping

        self.limits = limits
        self.fitness = None
        self.g_best = None
        self.g_best_coords = None
        self.trials = None

    def _before_initialization(self):
        super()._before_initialization()
        if not isinstance(self.limits, int) or self.limits < 1:
            raise ValueError("limits should be a positive integer")

    def initialize(self, problem_dict):
        super().initialize(problem_dict)
        self.g_best = np.inf if self._minmax() == min else -np.inf

        self.fitness = np.array([self.function(self.coords[i]) for i in range(self.population_size)])

        self.trials = np.zeros(self.population_size)

    def _coordinate_update_phase(self, i, k):
        phi = np.random.uniform(-1, 1, self.dimensions)
        new_coords = ne.evaluate("coords + phi * (coords - new_coords)", local_dict={'coords': self.coords[i],
                                                                                     'phi': phi,
                                                                                     'new_coords': self.coords[k]})
        new_coords = np.clip(new_coords, self.lb, self.ub)
        new_fitness = self.function(new_coords)
        if self._minmax()(np.array([self.fitness[i], new_fitness])) != self.fitness[i]:
            self.coords[i] = new_coords
            self.fitness[i] = new_fitness
            self.trials[i] = 0
        else:
            self.trials[i] += 1

    def _employed_bees_phase(self):
        for i in range(self.population_size):
            k = np.random.choice([j for j in range(self.population_size) if j != i])
            self._coordinate_update_phase(i, k)

    def _onlooker_bees_phase(self):
        if np.all(self.fitness == 0):
            probabilities = np.ones(self.population_size) / self.population_size
        else:
            probabilities = self.fitness / np.sum(self.fitness)
        for i in range(self.population_size):
            k = np.random.choice([j for j in range(self.population_size)], p=probabilities)
            self._coordinate_update_phase(i, k)

    def _scout_bees_phase(self):
        for i in range(self.population_size):
            if self.trials[i] > self.limits:
                self.coords[i] = np.random.uniform(self.lb, self.ub, self.dimensions)
                self.fitness[i] = self.function(self.coords[i])
                self.trials[i] = 0

    def evolve(self, epoch):
        self._employed_bees_phase()
        self._onlooker_bees_phase()
        self._scout_bees_phase()

        best_index = self._argminmax()(self.fitness)
        # if self._minmax()(np.array([self.fitness[best_index], self.g_best])) != self.g_best:
        #     self.g_best = self.fitness[best_index]
        #     self.g_best_coords = self.coords[best_index]
        self.g_best = self.fitness[best_index]
        self.g_best_coords = self.coords[best_index]

    def get_best_score(self):
        return self.g_best

    def get_best_solution(self):
        return self.g_best_coords

    def get_current_best_score(self):
        return self._minmax()(self.fitness)

    def get_current_best_solution(self):
        best_index = self._argminmax()(self.fitness)
        return self.coords[best_index]
