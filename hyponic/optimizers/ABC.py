from hyponic.optimizers.base_optimizer import BaseOptimizer
import numpy as np
from copy import deepcopy


class ABC(BaseOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.limits = kwargs.get('limits', 25)  # The number of trials before abandoning a food source
        self.coords = None
        self.fitness = None
        self.g_best = None
        self.g_best_coords = None
        self.trials = None

    def initialize(self, problem_dict):
        super().initialize(problem_dict)
        self.g_best = np.inf if self._minmax() == min else -np.inf

        self.coords = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimensions))
        self.fitness = np.array([self.function(self.coords[i]) for i in range(self.population_size)])

        self.trials = np.zeros(self.population_size)

    def _employed_bees_phase(self):
        for i in range(self.population_size):
            k = np.random.choice([j for j in range(self.population_size) if j != i])
            phi = np.random.uniform(-1, 1, self.dimensions)
            new_coords = self.coords[i] + phi * (self.coords[i] - self.coords[k])

            # TODO: if lb or ub is provided, clip the coordinates
            new_coords = np.clip(new_coords, self.lb, self.ub)
            new_fitness = self.function(new_coords)
            if self._minmax()(np.array([self.fitness[i], new_fitness])) != self.fitness[i]:
                self.coords[i] = new_coords
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def _onlooker_bees_phase(self):
        probabilities = self.fitness / np.sum(self.fitness)
        if np.isnan(probabilities).any():
            probabilities = np.ones(self.population_size) / self.population_size
        for i in range(self.population_size):
            k = np.random.choice([j for j in range(self.population_size)], p=probabilities)
            phi = np.random.uniform(-1, 1, self.dimensions)
            new_coords = self.coords[i] + phi * (self.coords[i] - self.coords[k])
            new_coords = np.clip(new_coords, self.lb, self.ub)
            new_fitness = self.function(new_coords)
            if self._minmax()(np.array([self.fitness[i], new_fitness])) != self.fitness[i]:
                self.coords[i] = new_coords
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                self.trials[i] += 1

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
        self.g_best = self.fitness[best_index]
        self.g_best_coords = deepcopy(self.coords[best_index])

    def get_best_score(self):
        return self.g_best

    def get_best_solution(self):
        return self.g_best_coords
