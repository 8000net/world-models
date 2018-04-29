from abc import ABC, abstractmethod

import numpy as np

class Solver(ABC):
    @abstractmethod
    def __init__(self):
        """
        Set population size, and any solver specific parameters
        """
        pass

    @abstractmethod
    def ask(self):
        """
        Return N possible solutions, where N = popsize
        """
        pass

    @abstractmethod
    def tell(self, fitness_list):
        """
        Tell solver fitness scores from each solution in population,
        update solver params based on best solution, and set
        best solution and best fitness score
        """
        pass

    @abstractmethod
    def result(self):
        """
        Return best solution and best fitness
        Should be called after tell
        """
        pass


class ES(Solver):
    """
    Sample solutions from a multivariate normal distribution
    with a mean and fixed standard deviation.
    After each generation, set mean to the solution with highest
    fitness score.
    """
    def __init__(self, pop_size, n_dim, stddev):
        self.pop_size = pop_size
        self.n_dim = n_dim
        self.stddev = stddev
        self.mean = np.zeros(n_dim)
        self.cov = np.diag(np.ones(n_dim))

    def ask(self):
        self.solutions = [np.random.multivariate_normal(self.mean, self.cov)
                          for _ in range(self.pop_size)]
        return self.solutions

    def tell(self, fitness_list):
        max_i = np.argmax(fitness_list)
        self.best_solution = self.solutions[max_i]
        self.best_fitness = fitness_list[max_i]
        self.mean = self.best_solution

    def result(self):
        return self.best_solution, self.best_fitness
