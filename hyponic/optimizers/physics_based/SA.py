from hyponic.optimizers.base_optimizer import BaseOptimizer

import numpy as np


class SA(BaseOptimizer):
    """
    Simulated Annealing
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.currents = None
        self.best = None
        self.best_score = None

    def initialize(self, problem_dict):
        super().initialize(problem_dict)

        self.currents = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimensions))

        self.best = self.currents[0]
        self.best_score = self.function(self.best)

    def evolve(self, current_epoch):
        progress = current_epoch / self.epoch
        t = max(0.01, min(1, 1 - progress))

        amplitudes = (np.max(self.intervals) - np.min(self.intervals)) * progress * 0.1
        deltas = np.random.uniform(-amplitudes / 2, amplitudes / 2, (self.population_size, self.dimensions))

        candidates = self.currents + deltas

        for idx, candidate in enumerate(candidates):
            candidate = np.clip(candidate, self.lb, self.ub)
            candidate_score = self.function(candidate)

            if candidate_score < self.best_score:
                self.best = candidate
                self.best_score = candidate_score
                self.currents[idx] = candidate
            else:
                score_abs_diff = abs(candidate_score - self.best_score)

                if np.random.uniform() < np.exp(-score_abs_diff / t):
                    self.currents[idx] = candidate

    def get_best_score(self):
        return self.best_score

    def get_best_solution(self):
        return self.best
