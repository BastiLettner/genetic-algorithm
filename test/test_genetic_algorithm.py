# The Genetic Algorithm, Crossover and Mutation

import unittest
from genetic_algorithm.chromosome import Chromosome
from genetic_algorithm.crossover import KPointCrossover
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.crossover import SinglePointCrossedUniformCrossover
from genetic_algorithm.crossover import PMCrossover, UPMCrossover, OrderedCrossover
from genetic_algorithm import population as population
from genetic_algorithm.fitness import AbstractFitnessStrategy
from genetic_algorithm.mutation import UniformMutation, RandomSwapMutation
from genetic_algorithm.selection import GreedySelection


class TestCrossover(unittest.TestCase):

    """ Test for the crossover strategies """

    def test_k_point_crossover(self):

        """ Testing the k point crossover """

        crossover_strategy = KPointCrossover(k_points=[3, 5, 15], problem_size=20)

        parent_one = Chromosome([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        parent_two = Chromosome([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        child_one, child_two = crossover_strategy.crossover(parent_one, parent_two)

        self.assertEqual(
            child_one.genetic_string,
            [0, 1, 2, 3, 15, 14, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3, 2, 1, 0],
            msg="Incorrect crossover"
        )
        self.assertEqual(
            child_two.genetic_string,
            [19, 18, 17, 16, 4, 5, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 16, 17, 18, 19],
            msg="Incorrect crossover"
        )

        # maximal amount of changes
        crossover_strategy = KPointCrossover(
            k_points=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            problem_size=20
        )

        parent_one = Chromosome([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        parent_two = Chromosome([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        child_one, child_two = crossover_strategy.crossover(parent_one, parent_two)

        self.assertEqual(
            child_one.genetic_string,
            [0, 18, 2, 16, 4, 14, 6, 12, 8, 10, 10, 8, 12, 6, 14, 4, 16, 2, 18, 0],
            msg="Incorrect crossover"
        )
        self.assertEqual(
            child_two.genetic_string,
            [19, 1, 17, 3, 15, 5, 13, 7, 11, 9, 9, 11, 7, 13, 5, 15, 3, 17, 1, 19],
            msg="Incorrect crossover"
        )

    def test_single_point_crossed_uniform_crossover(self):
        """ Test the SinglePointCrossedUniformCrossover strategy """

        cross_over_strategy = SinglePointCrossedUniformCrossover(problem_size=20, seed=0)

        # point = 13  obtained from the rng with seed 0

        parent_one = Chromosome([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        parent_two = Chromosome([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        child_one, child_two = cross_over_strategy.crossover(parent_one, parent_two)

        self.assertEqual(
            parent_one.genetic_string,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        )
        self.assertEqual(
            parent_two.genetic_string,
            [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        )
        self.assertEqual(
            child_one.genetic_string,
            [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14, 15, 16, 17, 18, 19]
        )
        self.assertEqual(
            child_two.genetic_string,
            [19, 18, 17, 16, 15, 14, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        )

    def test_pm_crossover(self):
        """ Test the Partially Mapped Crossover """

        cross_over_strategy = PMCrossover(problem_size=20, seed=10)

        parent_one = Chromosome([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        parent_two = Chromosome([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        child_one, child_two = cross_over_strategy.crossover(parent_one, parent_two)

        self.assertEqual(
            parent_one.genetic_string,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        )
        self.assertEqual(
            parent_two.genetic_string,
            [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        )

        self.assertEqual(len(child_one.genetic_string), len(set(child_one.genetic_string)), "The genetic code is not a "
                                                                                            "set anymore.")

        self.assertEqual(len(child_two.genetic_string), len(set(child_two.genetic_string)), "The genetic code is not a "
                                                                                            "set anymore.")

    def test_upm_crossover(self):
        """ Test Uniform Partially Mapped Crossover """

        cross_over_strategy = UPMCrossover(seed=0, problem_size=20, probability=.5)

        parent_one = Chromosome([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        parent_two = Chromosome([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        _, _ = cross_over_strategy.crossover(parent_one, parent_two)
        # check that the parents are unaltered
        self.assertEqual(
            parent_one.genetic_string,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        )
        self.assertEqual(
            parent_two.genetic_string,
            [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        )

    def test_ordered_crossover(self):
        """ Test ordered crossover. Should work with sets """

        cross_over_strategy = OrderedCrossover(seed=0, problem_size=20)

        parent_one = Chromosome([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        parent_two = Chromosome([19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

        child_one, child_two = cross_over_strategy.crossover(parent_one, parent_two)
        # check that the parents are unaltered
        self.assertEqual(
            parent_one.genetic_string,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        )
        self.assertEqual(
            parent_two.genetic_string,
            [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        )

        self.assertEqual(len(child_one.genetic_string), len(set(child_one.genetic_string)), "The genetic code is not a "
                                                                                            "set anymore.")

        self.assertEqual(len(child_two.genetic_string), len(set(child_two.genetic_string)), "The genetic code is not a "
                                                                                            "set anymore.")


class TestPopulation(unittest.TestCase):

    """ Testing the Population and initialization strategies """

    def setUp(self):

        self.population = population.Population(size=5)

        self.population.add_member(chromosome=Chromosome(genetic_code=[0, 1, 2]))
        self.population.add_member(chromosome=Chromosome(genetic_code=[1, 2, 0]))
        self.population.add_member(chromosome=Chromosome(genetic_code=[2, 1, 0]))
        self.population.add_member(chromosome=Chromosome(genetic_code=[3, 0, 0]))
        self.population.add_member(chromosome=Chromosome(genetic_code=[3, 1, 0]))
        self.population.members[0].fitness = 0
        self.population.members[1].fitness = 1
        self.population.members[2].fitness = 2
        self.population.members[3].fitness = 3
        self.population.members[4].fitness = 3

    def test_population_best_member(self):

        """ Test the Population class """

        master_of_the_universe = self.population.get_fittest_member()
        self.assertEqual(master_of_the_universe.genetic_string, [3, 0, 0], "Wrong chromosome returned ")

    def test_population_best_percentile(self):
        """ Test getting the best percentile """

        best_members = self.population.get_best_percentile(0.7)
        self.assertEqual(len(best_members), 3, "Two members should be returned")
        self.assertTrue(best_members[0].fitness > 1, msg="Wrong member returned")
        self.assertTrue(best_members[1].fitness > 1, msg="Wrong member returned")
        self.assertTrue(best_members[2].fitness > 1, msg="Wrong member returned")

    def test_random_initialization(self):
        """ Test the random initialization of the population """

        pop = population.RandomPermutedInitialization(size=5, problem_size=5, seed=0).generate_population()
        self.assertEqual(pop.num_members, 5, msg="Wrong number of member after generation with {}".format(
            population.RandomPermutedInitialization.__class__.__name__
        ))
        for member in pop.members:
            self.assertEqual(len(member.genetic_string), 5, msg="The population members have incorrect problem size")

    def test_population_best_n(self):
        """ Test to get the best n members """
        best_n = self.population.get_best_n(3)
        self.assertEqual(len(best_n), 3, "Two members should be returned")
        self.assertTrue(best_n[0].fitness > 1, msg="Wrong member returned")
        self.assertTrue(best_n[1].fitness > 1, msg="Wrong member returned")
        self.assertTrue(best_n[2].fitness > 1, msg="Wrong member returned")

    def test_population_uniform_initialization(self):
        """ Test the uniform initialization method """

        pop = population.RandomUniformInitialization(
            size=5,
            problem_size=10,
            lower=0,
            upper=5.0,
            seed=0
        ).generate_population()
        self.assertEqual(len(pop.members), 5, msg="Wrong population size after Generation with {}".format(
            population.RandomUniformInitialization.__class__.__name__
        ))
        for member in pop.members:
            self.assertEqual(
                len(member.genetic_string),
                10,
                "The length of the genetic string does not "
                "equal the problem size after population initialization"
                "with {}".format(population.RandomUniformInitialization.__class__.__name__)
            )
            for value in member.genetic_string:
                self.assertGreaterEqual(value, 0.0, "Wrong lower bound after population initialization with {}".format(
                    population.RandomUniformInitialization.__class__.__name__)
                )
                self.assertLessEqual(value, 5.0, "Wrong upper bound after population initialization with {}".format(
                    population.RandomUniformInitialization.__class__.__name__)
                )




class TestMutation(unittest.TestCase):

    """ Testing Mutation Strategies """

    def setUp(self):
        self.chromosome = Chromosome(genetic_code=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_uniform_mutation(self):
        """ Test Mutation with uniform mutated values """
        mutation_strategy = UniformMutation(low=10, high=11, n=2, seed=0, problem_size=10)  # force to insert 2 tens
        mutation_strategy.mutate(self.chromosome)

        self.assertNotEqual(self.chromosome.genetic_string, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], msg="Chromosome was not"
                                                                                                "mutated")
        self.assertTrue(10 in self.chromosome.genetic_string, msg="There should be a ten in the genetic string")
        self.chromosome.genetic_string.remove(10)  # remove the first ten
        self.assertTrue(10 in self.chromosome.genetic_string, msg="There should be a ten in the genetic string")
        self.chromosome.genetic_string.remove(10)  # remove the second ten
        self.assertFalse(10 in self.chromosome.genetic_string, msg="There should be no ten in the genetic string")

    def test_random_swap_mutation(self):
        """ Test the strategy where elements are randomly swapped """

        mutation_strategy = RandomSwapMutation(seed=0, n_swaps=2, problem_size=10, probability=1.0)
        mutation_strategy.mutate(self.chromosome)
        self.assertNotEqual(self.chromosome.genetic_string, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], msg="Chromosome was not"
                                                                                                "mutated")


class TestSelection(unittest.TestCase):

    """ Test the selection strategies """

    def setUp(self):

        self.population = population.Population(size=5)

        self.population.add_member(chromosome=Chromosome(genetic_code=[0, 1, 2]))
        self.population.add_member(chromosome=Chromosome(genetic_code=[1, 2, 0]))
        self.population.add_member(chromosome=Chromosome(genetic_code=[2, 1, 0]))
        self.population.add_member(chromosome=Chromosome(genetic_code=[3, 0, 0]))
        self.population.add_member(chromosome=Chromosome(genetic_code=[3, 1, 0]))
        self.population.members[0].fitness = 0
        self.population.members[1].fitness = 1
        self.population.members[2].fitness = 2
        self.population.members[3].fitness = 3
        self.population.members[4].fitness = 3

    def test_greedy_selection(self):
        """ Test the selection which uses the top x % of members for the next generation"""

        greedy_selection = GreedySelection(percentage_survivors=0.8)
        survivors = greedy_selection.select(self.population)
        self.assertEqual(len(survivors), 4, msg="There should be 4 survivors")
        for survivor in survivors:
            self.assertTrue(survivor.fitness > 0, "All survivors should have fitness larger than zero")


class TestGeneticAlgorithm(unittest.TestCase):

    """ Testing the genetic algorithm """

    def setUp(self):

        class SimpleFitness(AbstractFitnessStrategy):

            def fitness(self, chromosomes):
                for chromosome in chromosomes:
                    chromosome.fitness = chromosome.genetic_string[0]

        self.genetic_algorithm = GeneticAlgorithm(
            initial_population_strategy=population.RandomPermutedInitialization(size=10, problem_size=5, seed=0),
            cross_over_strategy=KPointCrossover(k_points=[3], problem_size=5),
            mutation_strategy=UniformMutation(low=0, high=5, n=1, seed=0, problem_size=5),
            fitness_strategy=SimpleFitness(),
            selection_strategy=GreedySelection(percentage_survivors=0.8),
            num_generations=10,
            seed=0
        )

    def test_get_parent_pairs(self):
        """ Test the function which yields parent pairs for cross over """
        pop = population.RandomPermutedInitialization(size=5, problem_size=10).generate_population()
        counter = 0
        for p1, p2 in self.genetic_algorithm.get_parent_pairs(pop):
            counter += 1
        self.assertEqual(counter, 2, msg="Wrong number of parent pairs returned")

    def test_calculate_population_fitness(self):
        """ Test the calculation of the fitness for each chromosome """
        pop = population.RandomPermutedInitialization(size=5, problem_size=10).generate_population()
        self.genetic_algorithm.calculate_population_fitness(pop)

        self.assertEqual(pop.members[0].fitness, pop.members[0].genetic_string[0])
        self.assertEqual(pop.members[1].fitness, pop.members[1].genetic_string[0])
        self.assertEqual(pop.members[2].fitness, pop.members[2].genetic_string[0])
        self.assertEqual(pop.members[3].fitness, pop.members[3].genetic_string[0])
        self.assertEqual(pop.members[4].fitness, pop.members[4].genetic_string[0])


if __name__ == '__main__':
    unittest.main()
