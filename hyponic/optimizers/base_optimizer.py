from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    """
    Base class for all optimizers. All optimizers should inherit from this class
    """

    def __init__(self, **kwargs):
        self.epoch = kwargs.get('epoch', 10)
        self.population_size = kwargs.get('population_size', 10)
        self.function = None
        self.lb = None
        self.ub = None
        self.dimensions = None
        self.minmax = kwargs.get('minmax', None)

    @abstractmethod
    def initialize(self, problem_dict):
        """
        Initialize the optimizer with the problem dictionary
        :param problem_dict: dictionary containing the problem definition
        :return: None
        """
        pass

    @abstractmethod
    def evolve(self):
        """
        Evolve the population for one epoch
        :param epoch: current epoch number
        :return: None
        """
        pass

    @abstractmethod
    def get_best_score(self):
        """
        Get the best score of the current population
        :return: best score
        """
        pass

    @abstractmethod
    def get_best_solution(self):
        """
        Get the best solution of the current population
        :return: best solution
        """
        pass

    def solve(self, problem_dict, verbose=False):
        """
        Solve the problem
        :param problem_dict: dictionary containing the problem definition
        :param verbose: if True, prints the best score at each epoch
        :return: None
        """
        self.initialize(problem_dict)
        for epoch in range(self.epoch):
            self.evolve()
            if verbose:
                print(f'Epoch: {epoch}, Best Score: {self.get_best_score()}')
        return self.get_best_solution(), self.get_best_score()
