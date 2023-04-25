from .base_optimizer import BaseOptimizer
import numpy as np
from copy import deepcopy


class PSO(BaseOptimizer):
    def __init__(self, **kwargs):
        self.epoch = kwargs.get('epoch', 10)
        self.population_size = kwargs.get('population_size', 10)
        self.a1 = kwargs.get('a1', 1)
        self.a2 = kwargs.get('a2', 1)
        self.function = None
        self.lb = None
        self.ub = None
        self.dimensions = None
        self.minmax = kwargs.get('minmax', None)
        self.coords = [0] * self.population_size
        self.velocities = [0] * self.population_size
        self.p_best = [0] * self.population_size
        self.p_best_coords = [0] * self.population_size
        self.g_best = None
        self.g_best_coords = None

    def _minmax(self):
        if self.minmax == 'min':
            return np.min
        return np.max

    def _argminmax(self):
        if self.minmax == 'min':
            return np.argmin
        return np.argmax

    def initialize(self, problem_dict):
        # TODO: add multithreading and multiprocessing
        self.function = problem_dict['fit_func']
        self.lb = problem_dict['lb']
        self.ub = problem_dict['ub']
        self.minmax = problem_dict['minmax'] if self.minmax is None else self.minmax
        self.dimensions = len(self.lb)
        for i in range(self.population_size):
            self.coords[i] = np.random.uniform(self.lb, self.ub, self.dimensions)
            self.velocities[i] = np.random.normal(size=self.dimensions)
            self.p_best[i] = self.function(self.coords[i])
            self.p_best_coords[i] = self.coords[i]
        self.g_best = self._minmax()(self.p_best)
        self.g_best_coords = self.p_best_coords[self._argminmax()(self.p_best)]

    def evolve(self, epoch):
        for i in range(self.population_size):
            self.velocities[i] = self.velocities[i] + self.a1 * np.random.random() * (
                    self.p_best[i] - self.coords[i]) + self.a2 * np.random.random() * (self.g_best - self.coords[i])
            temp_coords = self.coords[i] + self.velocities[i]
            if (temp_coords < self.lb).any() or (temp_coords > self.ub).any():
                continue
            self.coords[i] = deepcopy(temp_coords)
            if self._minmax()(np.concatenate([self.p_best[i], self.function(self.coords[i])])) != self.p_best[i]:
                self.p_best[i] = self.function(self.coords[i])
                self.p_best_coords[i] = deepcopy(self.coords[i])
        self.g_best = self._minmax()(self.p_best)
        # print(f"Epoch: {epoch}, Global best: {self.g_best}")
        self.g_best_coords = self.p_best_coords[self._argminmax()(self.p_best)]

    def get_best_score(self):
        return self.g_best

    def get_best_solution(self):
        return self.g_best_coords
