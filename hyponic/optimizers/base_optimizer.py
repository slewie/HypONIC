from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed



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
        if verbose:
            self.verbose = True
        self.initialize(problem_dict)
        for current_epoch in range(self.epoch):
            self.evolve(current_epoch)
            if self.verbose:
                print(f'Epoch: {current_epoch}, Best Score: {self.get_best_score()}')
        return self.get_best_solution(), self.get_best_score()
