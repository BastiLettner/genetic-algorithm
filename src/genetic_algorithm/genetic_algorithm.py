# Implements a genetic algorithm

import gin
import logging
import numpy as np
from .population import Population
from .population import AbstractInitialPopulationStrategy
from .selection import AbstractSelectionStrategy
from .mutation import AbstractMutationStrategy
from .crossover import AbstractCrossoverStrategy


@gin.configurable
class GeneticAlgorithm(object):

    """
    Implements a genetic algorithm.
    Uses a initial population of Chromosomes.
    An iterative process combines (crossover) and mutates those chromosomes to from new generations
    The chromosomes are selected to be in the next generation according to their fitness.

    Pseudo code:
        InitialisePopulation(population_size, problem_size)
        while not stop:
            parents = select_parents(fitness. population)
            new_generation = []
            for parent_1, parent_2 in parents:
                child_1, child_2 = crossover(parent_1, parent_2)
                new_generation.add(mutate(child_1))
                new_generation.add(mutate(child_2))
            population = new_generation
    """
    def __init__(
            self,
            initial_population_strategy,
            cross_over_strategy,
            mutation_strategy,
            fitness_strategy,
            selection_strategy,
            num_generations,
            seed=0
    ):
        """

        Args:
            initial_population_strategy(AbstractInitialPopulationStrategy): Initial Population Strategy. Defines how
                                                                            the population should be initialized.
            cross_over_strategy(AbstractCrossoverStrategy): The crossover strategy. Defines how two parent chromosomes
                                                            combine to child chromosomes.
            mutation_strategy(AbstractMutationStrategy): The mutation strategy. Defines how chromosomes are mutated.
            fitness_strategy(AbstractFitnessStrategy): The fitness function. Defines the fitness of chromosomes
            selection_strategy(AbstractSelectionStrategy): The strategy which defines the way parents are selected
            num_generations(int): The maximal number of generations
            seed(int): Seed for the rng.
        """
        self.initial_population = initial_population_strategy
        self.cross_over = cross_over_strategy
        self.mutation = mutation_strategy
        self.fitness_strategy = fitness_strategy
        self.selection = selection_strategy
        self.num_generations = num_generations
        self.rng = np.random.RandomState(seed=seed)
        self.scores = np.zeros(self.num_generations)
        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)
        self.logger.name = "GeneticAlgorithm"

    def train(self, hooks=None):
        """
        Execute the main loop of the Genetic Algorithm

        Args:
             hooks(list): List of hooks (functions) which have signature None = hook(GA, population, generation)

        Returns
            best_solution(Chromosome)
        """
        # generate the initial population
        population = self.initial_population.generate_population()

        # calculate the fitness of each member
        self.calculate_population_fitness(population)

        # get the best solution
        best_solution = population.get_fittest_member()

        logging.info(
            "Generation {} \t Best fitness {} \t Population Size {}".format(
                0,
                best_solution.fitness,
                population.num_members
            )
        )

        for generation in range(self.num_generations):

            children = []

            # select the breeders
            breeders = Population(size=population.size)
            breeders.members = self.selection.select(population)

            for p1, p2 in self.get_parent_pairs(breeders):

                c1, c2 = self.cross_over.crossover(p1, p2)

                self.mutation.mutate(c1)
                self.mutation.mutate(c2)

                children.append(c1)
                children.append(c2)

            # update the population
            population.members += children

            # recalculate the fitness of the new members
            self.calculate_population_fitness(population)

            # get the best and remain the population size
            population.members = population.get_best_n(population.size)

            # Get the best solution
            best_solution = population.get_fittest_member()

            self.logger.info(
                "Generation {} \t Best fitness {} \t Population Size {}".format(
                    generation + 1,
                    best_solution.fitness,
                    population.num_members
                )
            )

            self.scores[generation] = best_solution.fitness

            if hooks is not None:
                for hook in hooks:
                    hook(self, population, generation)

        return best_solution

    def get_parent_pairs(self, population):
        """
        Return pairs of Chromosomes ready for crossover.
        If the population size if and odd number one parent will be dropped.

        Args:
            population(Population): The population

        Returns:
            Python Generator
        """
        num_members = population.num_members
        idx = list(range(num_members))
        self.rng.shuffle(idx)
        for p1, p2 in zip(idx[::2], idx[1::2]):
            yield population.members[p1], population.members[p2]

    def calculate_population_fitness(self, population):
        """
        Calculate the fitness of each member.

        Args:
            population(Population): The current population

        Returns:

        """
        self.fitness_strategy.fitness(population.members)
