# Fitness of a chromosome

from abc import ABCMeta, abstractmethod


class AbstractFitnessStrategy(metaclass=ABCMeta):

    @abstractmethod
    def fitness(self, chromosomes):
        """
        Compute the fitness of a Chromosome
        This function sets the fitness attribute of the Chromosome

        Args:
            chromosomes(list): A list of chromosomes for which to calculate the fitness

        """
        raise NotImplementedError
