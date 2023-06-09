from hyponic.optimizers.base_optimizer import BaseOptimizer

import numpy as np
import numexpr as ne


class GA(BaseOptimizer):
    """
    Genetic Algorithm(GA)

    Example
    ~~~~~~~
        >>> from hyponic.optimizers.genetic_based.GA import GA
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
        >>> ga = GA(epoch=40, population_size=100, verbose=True, early_stopping=4)
        >>> ga.solve(problem_dict)
        >>> print(ga.get_best_score())
        >>> print(ga.get_best_solution())
    """

    def __init__(self, epoch: int = 10, population_size: int = 10, minmax: str = None,
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
        """
        super().__init__(**kwargs)
        self.epoch = epoch
        self.population_size = population_size
        self.minmax = minmax
        self.verbose = verbose
        self.mode = mode
        self.n_workers = n_workers
        self.early_stopping = early_stopping

        self.population = None
        self.scores = None
        self.best_score = None
        self.best_solution = None

    def initialize(self, problem_dict):
        super().initialize(problem_dict)

        self.population = np.random.uniform(low=self.lb, high=self.ub, size=(self.population_size, self.dimensions))
        self.scores = np.array([self.function(self.population[i]) for i in range(self.population_size)])

        best_idx = self._argminmax()(self.scores)
        self.best_score = self.scores[best_idx]
        self.best_solution = self.population[best_idx]

    def evolve(self, epoch):
        next_population = np.zeros_like(self.population)
        next_scores = np.zeros(self.population_size)

        # Elitism: keep the best solution from the previous generation
        best_idx = self._argminmax()(self.scores)
        next_population[0] = self.population[best_idx]
        next_scores[0] = self.scores[best_idx]

        # Roulette Wheel Selection
        fitness = self.scores - np.min(self.scores) if self.minmax == 'min' else np.max(self.scores) - self.scores
        total_fitness = np.sum(fitness)
        if total_fitness == 0:
            probs = np.ones(self.population_size) / self.population_size
        else:
            probs = fitness / total_fitness
        cum_probs = np.cumsum(probs)

        # Crossover and mutation
        for i in range(1, self.population_size):
            # Select two parents using roulette wheel selection
            parent1_idx = np.searchsorted(cum_probs, np.random.rand())
            parent2_idx = np.searchsorted(cum_probs, np.random.rand())

            # Single point crossover
            crossover_point = np.random.randint(1, self.dimensions)

            next_population[i, :crossover_point] = self.population[parent1_idx, :crossover_point]
            next_population[i, crossover_point:] = self.population[parent2_idx, crossover_point:]

            # Mutation
            mutation_strength = ne.evaluate("0.5 * (ub - lb)", local_dict={"ub": self.ub, "lb": self.lb})

            next_population[i] += np.random.normal(0, mutation_strength, size=self.dimensions)
            next_population[i] = np.clip(next_population[i], self.lb, self.ub)

        # evaluate the new population
        next_scores = np.array([self.function(next_population[i]) for i in range(self.population_size)])

        # update the best solution and score
        best_idx = self._argminmax()(next_scores)
        if self._minmax()(next_scores) < self._minmax()(self.scores):
            self.best_solution = next_population[best_idx]
            self.best_score = next_scores[best_idx]

        # replace the old population with the new one
        self.population = next_population
        self.scores = next_scores

    def get_best_score(self):
        return self.best_score

    def get_best_solution(self):
        return self.best_solution

    def get_current_best_score(self):
        return self._minmax()(self.scores)

    def get_current_best_solution(self):
        return self.population[self._argminmax()(self.scores)]
