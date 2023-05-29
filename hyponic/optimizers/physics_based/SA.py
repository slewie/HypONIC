from hyponic.optimizers.base_optimizer import BaseOptimizer

import numpy as np
import numexpr as ne


class SA(BaseOptimizer):
    """
    Simulated Annealing
    """

    def __init__(self, epoch: int = 10, population_size: int = 10, minmax: str = None,
                 verbose: bool = False, mode: str = 'single', n_workers: int = 4, early_stopping: int | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.population_size = population_size
        self.minmax = minmax
        self.verbose = verbose
        self.mode = mode
        self.n_workers = n_workers
        self.early_stopping = early_stopping

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

        amplitudes = ne.evaluate("(_max - _min) * progress * 0.1",
                                 local_dict={'_max': np.max(self.intervals), '_min': np.min(self.intervals),
                                             'progress': progress})
        deltas = np.random.uniform(-amplitudes / 2, amplitudes / 2, (self.population_size, self.dimensions))

        candidates = ne.evaluate("currents + deltas", local_dict={'currents': self.currents, 'deltas': deltas})

        for idx, candidate in enumerate(candidates):
            candidate = np.clip(candidate, self.lb, self.ub)
            candidate_score = self.function(candidate)

            if candidate_score < self.best_score:  # TODO: check if the problem is minimization or maximization
                self.best = candidate
                self.best_score = candidate_score
                self.currents[idx] = candidate
            else:
                score_abs_diff = ne.evaluate("abs(candidate_score - best_score)",
                                             local_dict={'candidate_score': candidate_score,
                                                         'best_score': self.best_score})

                if np.random.uniform() < np.exp(-score_abs_diff / t):
                    self.currents[idx] = candidate

    def get_best_score(self):
        return self.best_score

    def get_best_solution(self):
        return self.best

    def get_current_best_score(self):
        return self.function(self.currents[0])

    def get_current_best_solution(self):
        return self.currents[0]
