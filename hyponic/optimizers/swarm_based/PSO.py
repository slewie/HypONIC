from hyponic.optimizers.base_optimizer import BaseOptimizer

import numpy as np
from copy import deepcopy


class PSO(BaseOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a1 = kwargs.get('a1', 0.5)
        self.a2 = kwargs.get('a2', 0.5)
        self.coords = None
        self.velocities = None
        self.p_best = None
        self.p_best_coords = None
        self.g_best = None
        self.g_best_coords = None

    def initialize(self, problem_dict):
        # TODO: add multithreading and multiprocessing
        # TODO: check if the problem_dict is valid
        # TODO: if lb and ub are not provided, use the default values
        super().initialize(problem_dict)
        self.g_best = np.inf if self.minmax == "min" else -np.inf

        self.coords = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimensions))
        max_velocity = self.ub - self.lb

        self.velocities = np.random.uniform(-max_velocity, max_velocity, size=(self.population_size, self.dimensions))
        self.p_best_coords = deepcopy(self.coords)
        self.p_best = np.array([self.function(self.coords[i]) for i in range(self.population_size)])
        self._update_global_best()

    def evolve(self, epoch):
        self.velocities = self._update_velocity()
        self.coords = self.coords + self.velocities

        # TODO: if lb or ub is provided, clip the coordinates
        self.coords = np.clip(self.coords, self.lb, self.ub)
        fitness = np.array([self.function(self.coords[i]) for i in range(self.population_size)])
        condition = all(self._minmax()(np.concatenate([self.p_best, fitness])) != self.p_best)

        self.p_best_coords = np.where(condition, self.coords, self.p_best_coords)

        self.p_best = np.where(condition, fitness, self.p_best)
        self._update_global_best()

    def _update_velocity(self):
        r1 = np.random.random()
        r2 = np.random.random()
        return self.velocities + self.a1 * r1 * (self.p_best_coords - self.coords) + self.a2 * r2 * (self.g_best_coords - self.coords)

    def _update_global_best(self):
        if self._minmax()(np.concatenate([self.p_best, [self.g_best]])) != self.g_best:
            self.g_best = self._minmax()(self.p_best)
            self.g_best_coords = deepcopy(self.p_best_coords[self._argminmax()(self.p_best)])

    def get_best_score(self):
        return self.g_best

    def get_best_solution(self):
        return self.g_best_coords


class IWPSO(PSO):
    """
    Inertia Weight Particle Swarm Optimization
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = kwargs.get('w', 0.8)

    def _update_velocity(self):
        r1 = np.random.random()
        r2 = np.random.random()
        return self.w * self.velocities + self.a1 * r1 * (self.p_best_coords - self.coords) + self.a2 * r2 * (
                    self.g_best_coords - self.coords)
