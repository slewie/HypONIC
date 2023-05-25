from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from hyponic.utils.history import History
import time


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
        self.n_workers = kwargs.get('n_workers', 12)

        self.intervals = None
        self.dimensions = None
        self.verbose = kwargs.get('verbose', False)
        self.mode = kwargs.get('mode', 'single')
        self.coords = None

    def _before_initialization(self):
        """
        This method checks if the problem definition is correct
        """
        if not isinstance(self.epoch, int) or self.epoch < 1:
            raise ValueError("epoch should be a positive integer")

        if not isinstance(self.population_size, int) or self.population_size < 1:
            raise ValueError("population_size should be a positive integer")

        if self.mode not in ['single', 'multithread']:
            raise ValueError("mode should be either 'single' or 'multithread'")

        if self.n_workers < 1:  # TODO: n_workers can be -1, which means use all available cores
            raise ValueError("n_workers should be a positive integer")

    def _check_initialization(self):
        """
        This method checks if the problem definition in initialization function is correct
        """
        if self.lb is None or self.ub is None:
            raise ValueError("lb and ub should be provided")

        if not isinstance(self.lb, np.ndarray) or not isinstance(self.ub, np.ndarray):
            raise ValueError("lb and ub should be numpy arrays")

        if self.lb.shape != self.ub.shape:
            raise ValueError("lb and ub should have the same shape")

        if np.any(self.lb > self.ub):
            raise ValueError("lb should be less than ub")

        if self.minmax not in ['min', 'max']:
            raise ValueError("minmax should be either 'min' or 'max'")

        if self.function is None:
            raise ValueError("function should be provided")

        if not callable(self.function):
            raise ValueError("function should be callable")

    def initialize(self, problem_dict):
        """
        Initialize the optimizer with the problem dictionary
        :param problem_dict: dictionary containing the problem definition
        """
        # Unpack the problem dictionary
        self.minmax = problem_dict['minmax'] if self.minmax is None else self.minmax
        self.function = problem_dict["fit_func"]
        self.lb = np.array(problem_dict["lb"])
        self.ub = np.array(problem_dict["ub"])
        self._check_initialization()

        self.intervals = self.ub - self.lb
        self.dimensions = len(self.lb)

        # TODO: add flag that tracks history or not
        self.history = History(optimizer=self, epoch=self.epoch, population_size=self.population_size)

        # Create the population
        self.coords = self._create_population()

    def _create_individual(self):
        """
        Create an individual
        :return: individual
        """
        return np.random.uniform(self.lb, self.ub, self.dimensions)

    def _create_population(self):
        """
        Create a population of size self.population_size
        This method can be parallelized using multithreading or multiprocessing(but multiprocessing doesn't work now)

        :return: population
        """
        coords = np.zeros((self.population_size, self.dimensions))
        if self.mode == 'multithread':
            with ThreadPoolExecutor(self.n_workers) as executor:
                list_executor = [executor.submit(self._create_individual) for _ in range(self.population_size)]
                for f in as_completed(list_executor):
                    coords[list_executor.index(f)] = f.result()
        else:
            for i in range(self.population_size):
                coords[i] = self._create_individual()
        return coords

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

    @abstractmethod
    def get_current_best_score(self):
        """
        Get the best score of the current population
        :return: best score
        """
        pass

    @abstractmethod
    def get_current_best_solution(self):
        """
        Get the best solution of the current population
        :return: best solution
        """
        pass

    def _minmax(self):
        """
        Return the min or max function, depending on the minmax parameter
        """
        if self.minmax == 'min':
            return np.min
        return np.max

    def _argminmax(self):
        """
        Return the argmin or argmax function, depending on the minmax parameter
        """
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
        if verbose:
            self.verbose = True
        self._before_initialization()
        self.initialize(problem_dict)
        for current_epoch in range(self.epoch):
            start = time.time()
            self.evolve(current_epoch)
            end = time.time()
            if self.verbose:
                print(f'Epoch: {current_epoch}, Best Score: {self.get_best_score()}')
            self.history.update_history(current_epoch, end - start)
        return self.get_best_solution(), self.get_best_score()

    def get_history(self):
        return self.history.get_history()
