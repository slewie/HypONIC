import numpy as np

from hyponic.optimizers.swarm_based.ABC import ABC
from hyponic.optimizers.swarm_based.PSO import PSO, IWPSO


def sphere_function(x) -> np.ndarray:
    # min = 0 at x = [0, 0, 0, ...]
    return np.sum(np.square(x))


def rosenbrock_function(x):
    # min = 0 at x = [1, 1, 1, ...]
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin_function(x):
    # min = 0 at x = [0, 0, 0, ...]
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)


def griewank_function(x):
    # min = 0 at x = [0, 0, 0, ...]
    return np.sum(x ** 2 / 4000) - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1


def ackley_function(x):
    # min = 0 at x = [0, 0, 0, ...]
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / len(x))) - np.exp(
        np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.exp(1)


abc = ABC(epoch=40, population_size=100, minmax='min')
problem_dict = {
    'fit_func': ackley_function,
    'lb': [-5.12, -5, -14, -6, -0.9],
    'ub': [5.12, 5, 14, 6, 0.9],
    'minmax': 'min'
}
abc.solve(problem_dict)
print(abc.get_best_score())
print(abc.get_best_solution())