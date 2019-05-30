# Implements the strategy pattern for crossover methods

import gin
import numpy as np
import functools
import copy
from abc import ABCMeta, abstractmethod
from .chromosome import Chromosome


class AbstractCrossoverStrategy(metaclass=ABCMeta):

    """
    Implements an interface for Crossover methods
    """

    @abstractmethod
    def crossover(self, parent_one, parent_two):
        """
        Use two parent chromosomes to generate two child chromosomes.

        Args:
            parent_one(Chromosome): The first chromosome
            parent_two(Chromosome): The second chromosome

        Returns:
            child_one(Chromosome), child_two(Chromosome)
        """
        raise NotImplementedError


class SinglePointCrossover(AbstractCrossoverStrategy):

    def crossover(self, parent_one, parent_two):
        raise NotImplementedError


@gin.configurable
class KPointCrossover(AbstractCrossoverStrategy):

    """
    KPointCrossover is a crossover strategy where the two genetic strings are divided at k-points.
    The array parts between those points are then flipped between the parents.

    Example:
        ParentOne [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ParentTwo [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
                        |              |
        k_points: [1, 6]

        ChildOne [0, 1, 7, 6, 5, 4, 3, 8, 9]
        ChildTwo [9, 8, 2, 3, 4, 5, 6, 1, 0]
    """

    def __init__(self, k_points, problem_size):
        """
        Constructs object

        Args:
            k_points(list): list of indices at which the genetic string should be divided into the sub-arrays
                            The k-parameter has the following constrains:
                                - the maximal length is problem_size - 1
                                - the maximal value is problem_size - 2
            problem_size(int): The size of the problem, i.e. the length of the genetic string
        """
        assert len(k_points) < problem_size, "The k-point parameter must not contain more " \
                                             "values than problem size - 1"
        for point in k_points:
            assert point < problem_size - 1, "One of the k-points exceeds the maximal value of problem size - 2"

        self.k_points = sorted(k_points)
        self.problem_size = problem_size

    def crossover(self, parent_one, parent_two):
        """
        Apply the crossover.

        Args:
            parent_one(Chromosome): The first chromosome
            parent_two(Chromosome): The second chromosome

        Return:
            child_one(Chromosome), child_two(Chromosome)
        """
        assert self.problem_size == len(parent_one.genetic_string), "The size of the genetic string does not equal the" \
                                                                    "specified problem size"

        child_one = Chromosome(genetic_code=copy.copy(parent_one.genetic_string))
        child_two = Chromosome(genetic_code=copy.copy(parent_two.genetic_string))

        for i, range_start, range_end in zip(
                range(self.problem_size+1),
                [0] + self.k_points,
                self.k_points + [self.problem_size - 1]
        ):
            if i % 2 == 0:
                continue
            else:
                child_one.genetic_string[range_start+1:range_end+1] = parent_two.genetic_string[range_start+1:range_end+1]
                child_two.genetic_string[range_start+1:range_end+1] = parent_one.genetic_string[range_start+1:range_end+1]

        return child_one, child_two


@gin.configurable
class SinglePointCrossedUniformCrossover(AbstractCrossoverStrategy):

    """
    Like single point Crossover but the point is sampled from a uniform distribution
    In addition, the resulting sub-arrays are not copied into the same place but to the opposite side.

    Example:

        ParentOne [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ParentTwo [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        point = 4

        ChildOne [3, 2, 1, 0, 5, 3, 2, 1, 0]
        ChildTwo [0, 1, 2, 3, 4, 3, 2, 1, 0]



    """

    def __init__(self, problem_size, seed):
        """
        Construct Object
        Args:
            problem_size(int): Size of the Chromosomes genetic string
            seed(int): Seed for the rng
        """
        self.problem_size = problem_size
        self.rng = np.random.RandomState(seed=seed)
        self.sampler = functools.partial(self.rng.randint, low=1, high=problem_size)

    def crossover(self, parent_one, parent_two):
        """ Implement the crossover strategy """

        assert self.problem_size == len(parent_one.genetic_string), "The size of the genetic string does not equal the" \
                                                                    "specified problem size"

        child_one = Chromosome(genetic_code=copy.copy(parent_one.genetic_string))
        child_two = Chromosome(genetic_code=copy.copy(parent_two.genetic_string))

        point = self.sampler()
        r_point = self.problem_size - point

        child_one.genetic_string[0:point] = parent_two.genetic_string[r_point:]
        child_two.genetic_string[r_point:] = parent_one.genetic_string[0:point]

        return child_one, child_two


@gin.configurable
class PMCrossover(AbstractCrossoverStrategy):

    """ Implements partially mapped crossover. An important feature of PMX is that it works on problem
        where the genetic code must remain a set. For further information refer to
        http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/PMXCrossoverOperator.aspx
    """

    def __init__(self, seed, problem_size):
        """

        Args:
            seed(int): Seed for the rng
            problem_size(int): Size of the chromosomes genetic string
        """
        self.rng = np.random.RandomState(seed=seed)
        self.sampler_one = functools.partial(self.rng.randint, low=0, high=problem_size)
        self.sampler_two = functools.partial(self.rng.randint, low=0, high=problem_size-1)
        self.problem_size = problem_size

    def crossover(self, parent_one, parent_two):
        """
        Execute partially mapped crossover
        Args:
            parent_one(Chromosome): The first parent
            parent_two(Chromosome): The second parent

        Returns:
            child_one(Chromosome): The first parent
            child_two(Chromosome): The second parent
        """
        assert self.problem_size == len(parent_one.genetic_string), "The size of the genetic string does not equal the"\
                                                                    "specified problem size"

        parent_one_back_up = copy.copy(parent_one.genetic_string)
        parent_two_back_up = copy.copy(parent_two.genetic_string)
        child_one = Chromosome(genetic_code=[])
        child_two = Chromosome(genetic_code=[])

        p1, p2 = [0] * self.problem_size, [0] * self.problem_size

        # Initialize the position of each indices in the individuals
        for i in range(self.problem_size):
            p1[parent_one.genetic_string[i]] = i
            p2[parent_two.genetic_string[i]] = i
        # Choose crossover points
        cx_point1 = self.sampler_one()
        cx_point2 = self.sampler_two()
        if cx_point2 >= cx_point1:
            cx_point2 += 1
        else:  # Swap the two cx points
            cx_point1, cx_point2 = cx_point2, cx_point1

        # Apply crossover between cx points
        for i in range(cx_point1, cx_point2):
            # Keep track of the selected values
            temp1 = parent_one.genetic_string[i]
            temp2 = parent_two.genetic_string[i]
            # Swap the matched value
            parent_one.genetic_string[i], parent_one.genetic_string[p1[temp2]] = temp2, temp1
            parent_two.genetic_string[i], parent_two.genetic_string[p2[temp1]] = temp1, temp2
            # Position bookkeeping
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

        child_one.genetic_string = parent_one.genetic_string
        child_two.genetic_string = parent_two.genetic_string

        parent_one.genetic_string = parent_one_back_up
        parent_two.genetic_string = parent_two_back_up

        return child_one, child_two


@gin.configurable
class UPMCrossover(AbstractCrossoverStrategy):

    """
    Implements Uniform Partially Matched Crossover.
    Goes through all indices from each and inserts the value from the other parent with a certain probability
    This is done for both parents independently
    """
    def __init__(self, seed, problem_size, probability):
        """

        Args:
            seed(int): The seed for the rng
            problem_size(int): The size of the genetic code of each chromosome
            probability(float): Probability for each index to be switched
        """
        self.rng = np.random.RandomState(seed=seed)
        self.sampler = functools.partial(self.rng.rand)
        self.problem_size = problem_size
        self.probability = probability

    def crossover(self, parent_one, parent_two):
        """
        Executes the crossover strategy.
        Args:
            parent_one(Chromosome): The first parent
            parent_two(Chromosome): The second parent

        Returns:
            child_one(Chromosome): The first parent
            child_two(Chromosome): The second parent
        """
        assert self.problem_size == len(parent_one.genetic_string), "The size of the genetic string does not equal the"\
                                                                    "specified problem size"

        parent_one_back_up = copy.copy(parent_one.genetic_string)
        parent_two_back_up = copy.copy(parent_two.genetic_string)
        child_one = Chromosome(genetic_code=[])
        child_two = Chromosome(genetic_code=[])

        size = min(len(parent_one.genetic_string), len(parent_two.genetic_string))
        p1, p2 = [0] * size, [0] * size

        # Initialize the position of each indices in the individuals
        for i in range(size):
            p1[parent_one.genetic_string[i]] = i
            p2[parent_one.genetic_string[i]] = i

        for i in range(size):
            if self.sampler() < self.probability:
                # Keep track of the selected values
                temp1 = parent_one.genetic_string[i]
                temp2 = parent_two.genetic_string[i]
                # Swap the matched value
                parent_two.genetic_string[i], parent_one.genetic_string[p1[temp2]] = temp2, temp1
                parent_two.genetic_string[i], parent_two.genetic_string[p2[temp1]] = temp1, temp2
                # Position bookkeeping
                p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
                p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

        child_one.genetic_string = parent_one.genetic_string
        child_two.genetic_string = parent_two.genetic_string

        parent_one.genetic_string = parent_one_back_up
        parent_two.genetic_string = parent_two_back_up

        return child_one, child_two


@gin.configurable
class OrderedCrossover(AbstractCrossoverStrategy):

    """ Implements Ordered Crossover """

    def __init__(self, seed, problem_size):
        """

        Args:
            seed(int): Seed for the rng
            problem_size(int): The size of the genetic code of each chromosome
        """
        self.rng = np.random.RandomState(seed=seed)
        self.sampler = functools.partial(self.rng.choice, a=range(problem_size), size=2)
        self.problem_size = problem_size

    def crossover(self, parent_one, parent_two):
        """
        Args:
            parent_one(Chromosome): The first parent
            parent_two(Chromosome): The second parent

        Returns:
            child_one(Chromosome): The first parent
            child_two(Chromosome): The second parent
        """
        assert self.problem_size == len(parent_one.genetic_string), "The size of the genetic string does not equal the"\
                                                                    "specified problem size"

        parent_one_back_up = copy.copy(parent_one.genetic_string)
        parent_two_back_up = copy.copy(parent_two.genetic_string)
        child_one = Chromosome(genetic_code=[])
        child_two = Chromosome(genetic_code=[])

        a, b = self.sampler()
        if a > b:
            a, b = b, a

        holes1, holes2 = [True] * self.problem_size, [True] * self.problem_size
        for i in range(self.problem_size):
            if i < a or i > b:
                holes1[parent_two.genetic_string[i]] = False
                holes2[parent_one.genetic_string[i]] = False

        # We must keep the original values somewhere before scrambling everything
        temp1, temp2 = parent_one.genetic_string, parent_two.genetic_string
        k1, k2 = b + 1, b + 1
        for i in range(self.problem_size):
            if not holes1[temp1[(i + b + 1) % self.problem_size]]:
                parent_one.genetic_string[k1 % self.problem_size] = temp1[(i + b + 1) % self.problem_size]
                k1 += 1

            if not holes2[temp2[(i + b + 1) % self.problem_size]]:
                parent_two.genetic_string[k2 % self.problem_size] = temp2[(i + b + 1) % self.problem_size]
                k2 += 1

        # Swap the content between a and b (included)
        for i in range(a, b + 1):
            parent_one.genetic_string[i], parent_two.genetic_string[i] = \
                parent_two.genetic_string[i], parent_one.genetic_string[i]

        child_one.genetic_string = parent_one.genetic_string
        child_two.genetic_string = parent_two.genetic_string

        parent_one.genetic_string = parent_one_back_up
        parent_two.genetic_string = parent_two_back_up

        return child_one, child_two


