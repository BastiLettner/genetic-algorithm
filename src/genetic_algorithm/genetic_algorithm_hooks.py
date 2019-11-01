# Implements hooks which are called during the GAs training phase.

import abc


class AbstractGeneticAlgorithmHook(metaclass=abc.ABCMeta):

    """ Interface for GA hooks. """

    @abc.abstractmethod
    def hook(self, ga, population, generation):
        """
        Implemented in child class

        Args:
            ga(GeneticAlgorithm):
            population(list): The population. List of Chromosome objects.
            generation(int): The current generation of the GA

        Returns:
            None
        """
        raise NotImplementedError

    def __call__(self, ga, population, generation):
        self.hook(ga, population, generation)
