import numpy as np

from hyponic.optimizers.base_optimizer import BaseOptimizer


class GeneticAlgorithm(BaseOptimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.population = None
        self.scores = None
        self.best_score = None
        self.best_solution = None

    def initialize(self, problem_dict):
        super().initialize(problem_dict)

        self.population = np.random.uniform(low=self.lb, high=self.ub, size=(self.population_size, self.dimensions))
        self.scores = [self.function(x) for x in self.population]

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
            mutation_strength = 0.5 * (self.ub - self.lb)

            next_population[i] += np.random.normal(0, mutation_strength, size=self.dimensions)
            next_population[i] = np.clip(next_population[i], self.lb, self.ub)

        # evaluate the new population
        for i in range(self.population_size):
            next_scores[i] = self.function(next_population[i])

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
