# A single Chromosome


class Chromosome(object):

    """
    Represent a chromosome. Single member of the population.
    A chromosome consists of a genetic string, i.e. a list of numbers.

    """
    def __init__(self, genetic_code):
        """
        Abstract Constructor

        Args:
            genetic_code(list): List of elements representing the genetic code
        """
        self._genetic_code = genetic_code
        self._fitness = None

    @property
    def genetic_string(self):
        """

        Returns:
            genetic_string(list): List encoding the genetics of this chromosome
        """
        return self._genetic_code

    @genetic_string.setter
    def genetic_string(self, genetic_code):
        """

        Args:
            genetic_code(list): List encoding the genetics of this chromosome

        """
        self._genetic_code = genetic_code

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        """

        Args:
            fitness(float): Fitness of the member

        """
        self._fitness = fitness
