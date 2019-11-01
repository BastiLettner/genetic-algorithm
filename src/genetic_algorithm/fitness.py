# Fitness of a chromosome

import sys
import gin
import zmq
import logging
import multiprocessing as mp
from abc import ABCMeta, abstractmethod
from genetic_algorithm.chromosome import Chromosome


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


@gin.configurable
class MultiProcessingFitnessDecorator(object):

    """ Given a list of Chromosomes this implementation distributes the work using a zmq load balancer. """

    def __init__(
            self,
            fitness,
            n_workers=mp.cpu_count(),
            frontend="ipc://frontend",
            backend="ipc://backend",
            recalculate_all=True,
            log_level=logging.INFO
    ):
        """
        Initializes Fitness Calculator.

        Args:
            fitness(AbstractFitnessStrategy): An instance of the AbstractFitnessStrategy.
            n_workers(int): The number of processes to distribute the workload across
            frontend(str): The address the frontend binds on
            backend(str): The address the backend binds on
            recalculate_all(bool): Whether to calculate the fitness of all chromosomes passed to the fitness function
                                   (true) or only those with fitness not None (false). When a Chromosome has fitness
                                   not None it means it has not been altered since the last generation.
                                   In some cases the recalculation may be desired anyways.
            log_level(int): Log Level of the class
        """
        self._fitness = fitness
        self.__num_processes = n_workers
        self.frontend = frontend
        self.backend = backend
        self.recalculate_all = recalculate_all
        self.log_level = log_level
        self.broker, self.workers = self.__start_processes()

        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(self.frontend)
        self.logger = logging.getLogger("MultiProcessingFitness")
        self.logger.setLevel(log_level)

    def __start_processes(self):
        """ Start the broker and the n workers. The MultiProcessingFitness instance owns and manages the processes """
        # create processes
        broker_proc = mp.Process(target=self.__broker, args=[])
        worker_proc_list = []
        for i in range(self.__num_processes):
            worker_proc_list.append(
                mp.Process(target=self.__worker, args=[self._fitness, i])
            )
        # start processes
        broker_proc.start()
        for worker in worker_proc_list:
            worker.start()

        return broker_proc, worker_proc_list

    def __broker(self):
        """
        Proxy element. Intermediate between the client which receives the chromosomes and the Fitness Workers
        which process them and send back the result
        """
        context = zmq.Context()

        # Socket facing clients
        frontend = context.socket(zmq.ROUTER)
        frontend.bind(self.frontend)

        # Socket facing services
        backend = context.socket(zmq.DEALER)
        backend.bind(self.backend)

        zmq.proxy(frontend, backend)

        frontend.close()
        backend.close()
        context.term()

    def __worker(self, fitness, worker_id):
        """
        The worker process. Receives tasks through a socket. It calculates the fitness and returns the score

        Args:
            fitness(AbstractFitnessStrategy): The decorate fitness
        """

        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.connect(self.backend)
        logger = logging.getLogger("MultiProcessingFitness/Worker")
        logger.setLevel(self.log_level)
        formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        while True:
            message = socket.recv_json()
            logger.debug("Worker {} received message".format(worker_id))
            data = message["data"]
            task = message["task"]
            chromosome = Chromosome(genetic_code=data)
            fitness.fitness([chromosome])  # put single request into the fitness function
            response = {
                "task": task,
                "fitness": chromosome.fitness
            }
            socket.send_json(response)

    def terminate(self):
        """ Terminates all process and closes the sockets"""
        self.broker.terminate()
        for worker in self.workers:
            worker.terminate()
        self._socket.close()
        self._context.term()

    def fitness(self, chromosomes):
        """
        decorate the fitness function to load balance the request across processes

        Args:
            chromosomes(list): List of Chromosome instance containing the fitness
        """
        if not self.recalculate_all:
            # pick out the chromosomes with fitness None
            chromosomes = [chromosome for chromosome in chromosomes if chromosome.fitness is None]

        def completion(n_processed):
            return n_processed == len(chromosomes)

        num_processed = 0
        for i, chromosome in enumerate(chromosomes):
            self._socket.send_json(
                {
                    "task": i,
                    "data": chromosome.genetic_string
                }
            )
            ret = self._socket.recv_json(0.001)
            if ret is not None:
                chromosomes[ret["task"]].fitness = ret["fitness"]
                num_processed += 1
        while not completion(num_processed):
            ret = self._socket.recv_json()
            chromosomes[ret["task"]].fitness = ret["fitness"]

