from __future__ import annotations

import logging
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from mpi4py import MPI

from src.es.policy import Policy
from src.utils.TrainingResult import TrainingResult


class Reporter(ABC):
    @abstractmethod
    def start_gen(self):
        pass

    @abstractmethod
    def report_fits(self, fits: np.ndarray):
        pass

    @abstractmethod
    def report_noiseless(self, tr: TrainingResult, noiseless_policy: Policy):
        """Reports the fitness of a evaluation using no noise from the table and noiseless actions"""
        pass

    @abstractmethod
    def end_gen(self, time: float):
        pass


class MPIReporter(Reporter, ABC):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm

    def start_gen(self):
        if self.comm.rank == 0:
            self._start_gen()

    def report_fits(self, fits: np.ndarray):
        if self.comm.rank == 0:
            self._report_fits(fits)

    def report_noiseless(self, tr: TrainingResult, noiseless_policy: Policy):
        """Reports the fitness of a evaluation using no noise from the table and noiseless actions"""
        if self.comm.rank == 0:
            self._report_noiseless(tr, noiseless_policy)

    def end_gen(self, time: float):
        if self.comm.rank == 0:
            self._end_gen(time)

    @abstractmethod
    def _start_gen(self):
        pass

    @abstractmethod
    def _report_fits(self, fits: np.ndarray):
        pass

    @abstractmethod
    def _report_noiseless(self, tr: TrainingResult, noiseless_policy: Policy):
        """Reports the fitness of a evaluation using no noise from the table and noiseless actions"""
        pass

    @abstractmethod
    def _end_gen(self, time: float):
        pass


class StdoutReporter(MPIReporter):
    def _start_gen(self):
        pass

    def _report_fits(self, fits: np.ndarray):
        avg = np.mean(fits)
        mx = np.max(fits)
        print(f'avg:{avg:0.2f}-max:{mx:0.2f}')

    def _report_noiseless(self, tr: TrainingResult, noiseless_policy: Policy):
        print(f'noiseless:{tr.result[0]:0.2f}')

    def _end_gen(self, time: float):
        print(f'time {time:0.2f}')


class LoggerReporter(MPIReporter):
    def __init__(self, comm: MPI.Comm, cfg, log_name=None):
        super().__init__(comm)

        if comm.rank == 0:
            self.gen = 0
            self.cfg = cfg

            self.best_rew = 0
            self.best_dist = 0

            if log_name is None:
                log_name = datetime.now().strftime('es__%d_%m_%y__%H_%M_%S')
            logging.basicConfig(filename=f'logs/{log_name}.log', level=logging.DEBUG)
            logging.info('initialized logger')

    def _start_gen(self):
        logging.info(f'gen:{self.gen}')

    def _report_fits(self, fits: np.ndarray):
        logging.info(f'avg:{np.mean(fits):0.2f}')
        logging.info(f'max:{np.max(fits):0.2f}')

    def _report_noiseless(self, tr: TrainingResult, noiseless_policy: Policy):

        logging.info(f'noiseless fit:{tr.result}')
        # Calculating distance traveled (ignoring height dim). Assumes starting at 0, 0
        dist = np.linalg.norm(np.array(tr.behaviour[-3:-1]))
        rew = np.sum(tr.rewards)

        logging.info(f'dist: {dist}')
        logging.info(f'rew: {rew}')

        if rew > self.best_rew or dist > self.best_dist:
            self.best_rew = max(rew, self.best_rew)
            self.best_dist = max(dist, self.best_dist)
            folder = f'saved/{self.cfg.general.name}'
            if not os.path.exists(folder):
                os.makedirs(folder)
            pickle.dump(noiseless_policy, open(f'{folder}/policy-{self.gen}', 'wb'))

    def _end_gen(self, time: float):
        logging.info(f'time:{time:0.2f}')
        self.gen += 1
