from __future__ import annotations

import logging
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from mpi4py import MPI

from es.policy import Policy


class Reporter(ABC):
    @abstractmethod
    def start_gen(self, gen: int):
        pass

    @abstractmethod
    def report_fits(self, fits: np.ndarray):
        pass

    @abstractmethod
    def report_noiseless(self, fit: float):
        """Reports the fitness of a evaluation using no noise from the table and noiseless actions"""
        pass

    @abstractmethod
    def end_gen(self, time: float, policy: Policy):
        pass


class MPIReporter(Reporter, ABC):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm

    def start_gen(self, gen: int):
        if self.comm.rank == 0:
            self._start_gen(gen)

    def report_fits(self, fits: np.ndarray):
        if self.comm.rank == 0:
            self._report_fits(fits)

    def report_noiseless(self, fit: float):
        """Reports the fitness of a evaluation using no noise from the table and noiseless actions"""
        if self.comm.rank == 0:
            self._report_noiseless(fit)

    def end_gen(self, time: float, policy: Policy):
        if self.comm.rank == 0:
            self._end_gen(time, policy)

    @abstractmethod
    def _start_gen(self, gen: int):
        pass

    @abstractmethod
    def _report_fits(self, fits: np.ndarray):
        pass

    @abstractmethod
    def _report_noiseless(self, fit: float):
        """Reports the fitness of a evaluation using no noise from the table and noiseless actions"""
        pass

    @abstractmethod
    def _end_gen(self, time: float, noiseless_policy: Policy):
        pass


class StdoutReporter(MPIReporter):
    def _start_gen(self, gen: int):
        print(f'Gen:{gen}')

    def _report_fits(self, fits: np.ndarray):
        avg = np.mean(fits)
        mx = np.max(fits)
        print(f'avg:{avg:0.2f}-max:{mx:0.2f}')

    def _report_noiseless(self, fit: float):
        print(f'noiseless:{fit:0.2f}')

    def _end_gen(self, time: float, noiseless_policy: Policy):
        print(f'time {time:0.2f}')


class LoggerReporter(MPIReporter):
    def __init__(self, comm: MPI.Comm, cfg, log_name=None):
        super().__init__(comm)

        if comm.rank == 0:
            self.gen = 0
            self.cfg = cfg

            if log_name is None:
                log_name = datetime.now().strftime('es__%d_%m_%y__%H_%M_%S')
            logging.basicConfig(filename=f'logs/{log_name}.log', level=logging.DEBUG)
            logging.info('initialized logger')

    def _start_gen(self, gen: int):
        logging.info(f'gen:{gen}')
        self.gen = gen

    def _report_fits(self, fits: np.ndarray):
        logging.info(f'avg:{np.mean(fits):0.2f}')
        logging.info(f'max:{np.max(fits):0.2f}')

    def _report_noiseless(self, fit: float):
        logging.info(f'noiseless:{fit:0.2f}')

    def _end_gen(self, time: float, noiseless_policy: Policy):
        logging.info(f'time:{time:0.2f}')

        if self.gen % self.cfg.general.save_interval == 0 and self.cfg.general.save_interval > 0:  # checkpoints
            if not os.path.exists('saved'):
                os.makedirs('saved')
            pickle.dump(noiseless_policy, open(f'saved/policy-{self.gen}', 'wb'))
