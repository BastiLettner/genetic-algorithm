# Selection describes the process of finding a subset of chromosomes used for breeding a new Generation

import gin
from abc import ABCMeta, abstractmethod


class AbstractSelectionStrategy(metaclass=ABCMeta):

    """ Implements interface for selection strategies """

    @abstractmethod
    def select(self, population):
        """
        Select a subset of survivors out of a list of Chromosomes

        Args:
             population(Population): The current population

        Returns
            survivors(list): List of Chromosomes.
        """
        raise NotImplementedError


@gin.configurable
class GreedySelection(AbstractSelectionStrategy):

    """ Select the candidates for the new generation by picking the top ones """

    def __init__(self, percentage_survivors):
        """
        Constructs object
        Args:
            percentage_survivors(float): The percentage of chromosomes to use for the next generation
        """
        self.percentage_survivors = percentage_survivors

    def select(self, population):
        """
        Select a subset of survivors out of a list of Chromosomes
        This function assumes that the fitness measure of the chromosome is up to date

        Args:
            population(Population): The current population

        Returns
            survivors(list): List of Chromosomes.
        """
        return population.get_best_percentile(self.percentage_survivors)