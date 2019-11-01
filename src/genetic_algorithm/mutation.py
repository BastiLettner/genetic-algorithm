# implements mutation strategies of Chromosomes for a Genetic Algorithm

import gin
import numpy as np
import functools
from abc import ABCMeta, abstractmethod


class AbstractMutationStrategy(metaclass=ABCMeta):

    """
    Interface for mutation strategies for Chromosomes
    """
    def __init__(self, probability):
        """
        Constructor
        Args:
             probability(float): In range 0...1 : Chance of applying the mutation
        """
        self.probability = probability

    @abstractmethod
    def mutate(self, chromosome):
        """
        Mutate a chromosome according to the mutation strategy.

        Args:
            chromosome(Chromosome): The chromosome you want to mutate.

        Returns:
            None, the operation should be applied in-place

        """
        raise NotImplementedError


@gin.configurable
class UniformMutation(AbstractMutationStrategy):

    """
    Change n values in the Chromosomes genetic string with uniformly sampled values in a given range
    """
    def __init__(self, low, high, n, seed, problem_size, probability=1.0):
        """
        Construct Strategy
        Args:
            low(int): Lower bound for the uniform sampling
            high(int): Upper bound for the uniform sampling (>= low)
            n(int): The number of values to mutate on each mutation. Must be smaller or equal to the size of the genetic
                    string of a Chromosome
            seed(int): Seed for the rng
            problem_size(int): The length of the Chromosomes genetic string
            probability(float): In range 0...1 : Chance of applying the mutation
        """
        super(UniformMutation, self).__init__(probability=probability)
        self.random = np.random.RandomState(seed=seed)
        self.uniform_rng_values = functools.partial(self.random.randint, low=low, high=high)
        self.uniform_rng_indices = functools.partial(self.random.randint, low=0, high=problem_size)
        self.num_values_to_change = n

    def mutate(self, chromosome):
        """
        Randomly modify a chromosomes genetic String

        Args:
            chromosome(Chromosome): The Chromosome

        Returns:

        """
        if self.random.rand() < self.probability:
            chromosome.fitness = None  # Every time we change a chromosome we set its fitness to None
            for index in self.uniform_rng_indices(size=self.num_values_to_change):
                chromosome.genetic_string[index] = self.uniform_rng_values()


@gin.configurable
class RandomSwapMutation(AbstractMutationStrategy):

    """ Randomly swap values of the genetic code """

    def __init__(self, seed, n_swaps, problem_size, probability):
        """
        Constructor

        Args:
            seed(int): The seed for the rng
            n_swaps(int): The number of swaps
            problem_size(int): The size of the genetic string
            probability(float): probability of the mutation to be applied
        """
        super(RandomSwapMutation, self).__init__(probability=probability)
        self.rng = np.random.RandomState(seed=seed)
        self.sampler = functools.partial(self.rng.randint, low=0, high=problem_size)
        self.n_swaps = n_swaps

    def mutate(self, chromosome):
        """
        Mutate Chromosome by swapping n values

        Args:
            chromosome:

        Returns:

        """
        if self.rng.rand() < self.probability:
            chromosome.fitness = None
            for swap in range(self.n_swaps):
                i, j = self.sampler(), self.sampler()
                tmp = chromosome.genetic_string[i]
                chromosome.genetic_string[i] = chromosome.genetic_string[j]
                chromosome.genetic_string[j] = tmp


@gin.configurable
class WhiteNoiseMutation(AbstractMutationStrategy):

    """ Adds random normal distributed noise to values """

    def __init__(self, n, problem_size, seed=0, mean=0, std=1, probability=0.5):
        """
        Constructor

        Args:
            n(int): Number of values to add noise to
            problem_size(int): Size of the genetic string
            seed(int): The seed for the rng
            mean(float): mean of the normal dist
            std(float): standard deviation of the normal dist
            probability(float): probability of the mutation to be applied
        """
        super(WhiteNoiseMutation, self).__init__(probability=probability)
        self.rng = np.random.RandomState(seed=seed)
        self.sampler = functools.partial(self.rng.normal, loc=mean, scale=std)
        self.n = n
        self.problem_size = problem_size

    def mutate(self, chromosome):
        """
        Mutate Chromosome by swapping n values

        Args:
            chromosome:

        Returns:

        """
        if self.rng.rand() < self.probability:
            chromosome.fitness = None
            idx = self.rng.randint(0, self.problem_size, self.n)
            for i in idx:
                chromosome.genetic_string[i] = self.sampler() + chromosome.genetic_string[i]
