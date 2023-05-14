from hyponic.optimizers.base_optimizer import BaseOptimizer

import numpy as np
import numexpr as ne


class PSO(BaseOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a1 = kwargs.get('a1', 0.5)
        self.a2 = kwargs.get('a2', 0.5)
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
        # TODO: check if the problem_dict is valid
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


class IWPSO(PSO):
    """
    Inertia Weight Particle Swarm Optimization
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = kwargs.get('w', 0.8)

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
