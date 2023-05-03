import numpy as np

from hyponic.optimizers.base_optimizer import BaseOptimizer


class SA(BaseOptimizer):
    """
    Simulated Annealing
    """

    def __init__(self, T, **kwargs):
        super().__init__(**kwargs)

        self.T = T

        self.current = None
        self.best = None
        self.best_score = None

        self.step_size = 0.1

    def initialize(self, problem_dict):
        super().initialize(problem_dict)

        self.current = np.random.uniform(self.lb, self.ub, self.dimensions)

        self.best = self.current
        self.best_score = self.function(self.best)

    def evolve(self):
        t = self.T

        for i in range(self.epoch):
            candidate = self.current + np.random.uniform(-self.step_size, self.step_size, self.dimensions)
            candidate = np.clip(candidate, self.lb, self.ub)

            candidate_score = self.function(candidate)

            if candidate_score < self.best_score:
                self.best = candidate
                self.best_score = candidate_score
                self.current = candidate
            else:
                score_abs_diff = abs(candidate_score - self.best_score)

                if np.random.uniform() < np.exp(-score_abs_diff / self.T):
                    self.current = candidate

            t = t / (1 + i)

    def get_best_score(self):
        return self.best_score

    def get_best_solution(self):
        return self.best