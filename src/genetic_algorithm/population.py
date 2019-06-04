# Strategies for initializing populations

import functools
import gin
import numpy as np
from abc import ABCMeta, abstractmethod
from genetic_algorithm.chromosome import Chromosome


class Population(object):

    """ Represent the population """

    def __init__(self, size):
        """
        Construct Population

        Args:
            size(int): Size of the population.
        """
        self.size = size
        self.members = []
        self.__iter_ptr = 0

    def add_member(self, chromosome):

        """
        Add a member to the population. Only possible is population is not full yet

        Args:
            chromosome(Chromosome): A new chromosome

        Returns:

        """
        if len(self.members) == self.size:
            raise ValueError("Population Full")
        else:
            self.members.append(chromosome)

    def get_fittest_member(self):
        """
        Get the best (highest fitness) chromosome of the population
        If there a multiple solutions with the same fitness only one is returned

        Returns:
            Chromosome
        """
        scores = [chromosome.fitness for chromosome in self.members]
        return self.members[scores.index(max(scores))]

    def get_best_percentile(self, percentile):
        """
        Find the best x % from the current population and return their indices
        Args:
            percentile(float): (0, 1)

        Returns:
            list of Chromosomes
        """

        k_largest = int(percentile * len(self.members))
        scores = [chromosome.fitness for chromosome in self.members]
        best_percentile = []
        while len(best_percentile) < k_largest:
            max_idx = np.argmax(scores)
            best_percentile.append(self.members[max_idx])
            scores[max_idx] = -np.inf  # make sure this one is not selected again
        return best_percentile

    def get_best_n(self, n):
        """
        Get then best members
        Args:
            n(int): The number of members to return

        Returns:
            best_n(list): List of Chromosomes
        """
        assert n <= self.num_members, "Cannot get more members than the population holds"
        best_n = []
        scores = [chromosome.fitness for chromosome in self.members]
        while len(best_n) < n:
            max_idx = np.argmax(scores)
            best_n.append(self.members[max_idx])
            scores[max_idx] = -np.inf  # make sure this one is not selected again
        return best_n

    @property
    def num_members(self):
        """

        Returns:
            the number of members in the current population
        """
        return len(self.members)


class AbstractInitialPopulationStrategy(metaclass=ABCMeta):

    """ Implements interface for population initialization strategies """

    def __init__(self, size, problem_size):
        """
        Construct object
        Args:
            size(int): Size of the population
            problem_size(int): Size of the genetic string of each chromosome
        """
        self.size = size
        self.problem_size = problem_size

    @abstractmethod
    def generate_population(self):
        """
        Generate the initial population

        Returns:
            population(Population)
        """
        raise NotImplementedError


@gin.configurable
class RandomPermutedInitialization(AbstractInitialPopulationStrategy):

    """
    Initialize each chromosome with a random permutation of the number in range(problem_size)

    Example:
        problem_size = 10
        population_sample = [1, 5, 0, 8, 9, 4, 2, 7, 3, 6] (genetic_string)
    """

    def __init__(self, size, problem_size, seed=0):
        """
        Construct object
        Args:
            size(int): Size of the population
            problem_size(int): Size of the genetic string of each chromosome
        """
        super(RandomPermutedInitialization, self).__init__(size=size, problem_size=problem_size)
        self.random = np.random.RandomState(seed=seed)

    def generate_population(self):
        """
        Generate the initial population

        Returns:
            population(Population)
        """
        population = Population(size=self.size)
        for i in range(self.size):
            code = list(range(self.problem_size))
            self.random.shuffle(code)
            population.add_member(
                Chromosome(
                    genetic_code=list(code)
                )
            )
        return population


@gin.configurable
class RandomUniformInitialization(AbstractInitialPopulationStrategy):

    """ Initialize the genetic string with uniformly sampled value from range """

    def __init__(self, size, problem_size, lower, upper, seed=0):
        """
        Construct Object
        Args:
            size(int): Size of the population
            problem_size(int): Size of the problem. I.e. length of the genetic string
            seed(int): Seed for the rng
        """
        super(RandomUniformInitialization, self).__init__(
            size=size,
            problem_size=problem_size
        )
        self.rng = np.random.RandomState(seed=seed)
        self.sampler = functools.partial(self.rng.uniform, low=lower, high=upper)

    def generate_population(self):
        """
        Generate size population members and return the population
        Returns:
            pop(Population)
        """
        population = Population(size=self.size)
        for i in range(self.size):
            code = [self.sampler() for _ in range(self.problem_size)]
            population.add_member(
                Chromosome(
                    genetic_code=code
                )
            )

        return population


@gin.configurable
class RandomValueInitialization(AbstractInitialPopulationStrategy):

    """ Given a list of values the chromosome will use these values to initialize """

    def __init__(self, size, problem_size, values, seed=0):
        """
        Construct Object
        Args:
            size(int): Size of the population
            problem_size(int): Size of the problem. I.e. length of the genetic string
            seed(int): Seed for the rng
            values(list): List of values to choose the from. E.g. [0, 1] produces binary chromosomes
        """
        super(RandomValueInitialization, self).__init__(
            size=size,
            problem_size=problem_size
        )
        self.rng = np.random.RandomState(seed=seed)
        self.values = values

    def generate_population(self):
        """
        Generate size population members and return the population
        Returns:
            pop(Population)
        """
        population = Population(size=self.size)
        for i in range(self.size):
            code = [self.rng.choice(self.values) for _ in range(self.problem_size)]
            population.add_member(
                Chromosome(
                    genetic_code=code
                )
            )

        return population
