from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    """
    Base class for all optimizers. All optimizers should inherit from this class
    """

    def __init__(self, **kwargs):
        self.epoch = kwargs.get("epoch", 10)
        self.population_size = kwargs.get("population_size", 10)

        self.function = None
        self.lb = None
        self.ub = None
        self.minmax = kwargs.get('minmax', None)

        self.interval_len = None
        self.dimensions = None

    @abstractmethod
    def initialize(self, problem_dict):
        """
        Initialize the optimizer with the problem dictionary
        :param problem_dict: dictionary containing the problem definition
        :return: None
        """
        # Unpack the problem dictionary
        self.function = problem_dict["fit_func"]
        self.lb = np.array(problem_dict["lb"])
        self.ub = np.array(problem_dict["ub"])
        self.minmax = problem_dict['minmax'] if self.minmax is None else self.minmax

        self.intervals = self.ub - self.lb
        self.dimensions = len(self.lb)

    @abstractmethod
    def evolve(self, current_epoch):
        """
        Evolve the population for one epoch
        :param current_epoch: current epoch number
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

    def _minmax(self):
        if self.minmax == 'min':
            return np.min
        return np.max

    def _argminmax(self):
        if self.minmax == 'min':
            return np.argmin
        return np.argmax

    def solve(self, problem_dict, verbose=False):
        """
        Solve the problem
        :param problem_dict: dictionary containing the problem definition
        :param verbose: if True, prints the best score at each epoch
        :return: None
        """
        self.initialize(problem_dict)
        for current_epoch in range(self.epoch):
            self.evolve(current_epoch)
            if verbose:
                print(f'Epoch: {current_epoch}, Best Score: {self.get_best_score()}')
        return self.get_best_solution(), self.get_best_score()
