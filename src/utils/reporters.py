from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from os import path
from typing import Tuple, Dict

import numpy as np
from mlflow import log_params, log_metrics, set_experiment, start_run
from mpi4py import MPI
from munch import unmunchify, Munch
from pandas import json_normalize

from src.core.policy import Policy
from src.gym.training_result import TrainingResult


def calc_dist_rew(tr: TrainingResult) -> Tuple[float, float]:
    # Calculating distance traveled (ignoring height dim). Assumes starting at 0, 0
    return np.linalg.norm(np.array(tr.positions[-3:-1])), np.sum(tr.rewards)


class Reporter(ABC):
    """Absolute base class reporter, should rather subclass MpiReporter"""

    @abstractmethod
    def start_gen(self):
        pass

    @abstractmethod
    def log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, policy: Policy, steps: int):
        pass

    @abstractmethod
    def end_gen(self):
        pass

    @abstractmethod
    def print(self, s: str):
        """For printing one time information"""
        pass

    @abstractmethod
    def log(self, d: Dict[str, float]):
        """For logging key value pairs that recur each generation"""
        pass


class ReporterSet(Reporter):
    def __init__(self, *reporters: Reporter):
        self.reporters = [reporter for reporter in reporters if reporter is not None]

    def start_gen(self):
        for reporter in self.reporters:
            reporter.start_gen()

    def log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, policy: Policy, steps: int):
        for reporter in self.reporters:
            reporter.log_gen(fits, noiseless_tr, policy, steps)

    def end_gen(self):
        for reporter in self.reporters:
            reporter.end_gen()

    def print(self, s: str):
        for reporter in self.reporters:
            reporter.print(s)

    def log(self, d: Dict[str, float]):
        for reporter in self.reporters:
            reporter.log(d)


class MpiReporter(Reporter, ABC):
    """Thread safe reporter"""
    MAIN = 0

    def __init__(self, comm: MPI.Comm):
        self.comm = comm

    def start_gen(self):
        if self.comm.rank == MpiReporter.MAIN:
            self._start_gen()

    def log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, policy: Policy, steps: int):
        if self.comm.rank == MpiReporter.MAIN:
            self._log_gen(fits, noiseless_tr, policy, steps)

    def end_gen(self):
        if self.comm.rank == MpiReporter.MAIN:
            self._end_gen()

    def print(self, s: str):
        if self.comm.rank == MpiReporter.MAIN:
            self._print(s)

    def log(self, d: Dict[str, float]):
        if self.comm.rank == MpiReporter.MAIN:
            self._log(d)

    @abstractmethod
    def _start_gen(self):
        pass

    @abstractmethod
    def _log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, policy: Policy, steps: int):
        pass

    @abstractmethod
    def _end_gen(self):
        pass

    @abstractmethod
    def _print(self, s: str):
        pass

    @abstractmethod
    def _log(self, d: Dict[str, float]):
        pass


class DefaultMpiReporter(MpiReporter, ABC):
    """Useful information logged in log gen. To use this class create a subclass and override the _log function to use
    your printing method of choice."""

    def __init__(self, comm: MPI.Comm):
        super().__init__(comm)
        self.gen = 0
        self.cum_steps = 0
        self.gen_start_time = 0

    def _start_gen(self):
        self.gen_start_time = time.time()
        self.print('\n\n----------------------------------------')
        self.log({'gen': self.gen})

    def _log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, policy: Policy, steps: int):
        for i, col in enumerate(fits.T):
            # Objectives are grouped by column so this finds the avg and max of each objective
            self.log({f'avg-{i}': np.mean(col).round(2).item()})
            self.log({f'max-{i}': np.max(col).round(2).item()})

        self.cum_steps += steps
        dist, rew = calc_dist_rew(noiseless_tr)

        self.log({'dist': dist})
        self.log({'rew': rew})

        self.print('')
        self.log({f'steps': steps})
        self.log({f'cum steps': self.cum_steps})
        self.log({'n fits ranked': len(fits)})

    def _end_gen(self):
        self.log({'time': round(time.time() - self.gen_start_time, 2)})
        self.gen += 1


class DefaultMpiReporterSet(DefaultMpiReporter):
    def __init__(self, comm: MPI.Comm, run_name, *reporters: Reporter):
        super().__init__(comm)

        self.fit_folder = path.join('saved', run_name, 'fits')
        self.policy_folder = path.join('saved', run_name, 'weights')
        if comm.rank == MpiReporter.MAIN:
            if not path.exists(self.fit_folder): os.makedirs(self.fit_folder)
            if not path.exists(self.policy_folder): os.makedirs(self.policy_folder)

        self.reporters = [reporter for reporter in reporters if reporter is not None]

        self.best_rew = 0
        self.best_dist = 0

    def _log_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, policy: Policy, steps: int):
        super()._log_gen(fits, noiseless_tr, policy, steps)
        if self.comm.rank == MpiReporter.MAIN:  # saving policy and all fits to files
            dist, rew = calc_dist_rew(noiseless_tr)
            save_policy = (rew > self.best_rew or dist > self.best_dist)
            self.best_rew = max(rew, self.best_rew)
            self.best_dist = max(dist, self.best_dist)
            if save_policy:  # Saving policy if it obtained a better reward or distance
                policy.save(self.policy_folder, str(self.gen))
                self.print(f'saving policy with rew:{rew:0.2f} and dist:{dist:0.2f}')

            np.save(path.join(f'{self.fit_folder}', f'{self.gen}.np'), fits)

    def _log(self, d: Dict[str, float]):
        for reporter in self.reporters:
            reporter.log(d)

    def _print(self, s: str):
        for reporter in self.reporters:
            reporter.print(s)


class StdoutReporter(DefaultMpiReporter):
    def __init__(self, comm: MPI.Comm):
        super().__init__(comm)

    def _print(self, s: str):
        print(s)

    def _log(self, d: Dict[str, float]):
        for k, v in d.items():
            print(f'{k}:{v}')


class LoggerReporter(DefaultMpiReporter):
    def __init__(self, comm: MPI.Comm, log_folder=None):
        super().__init__(comm)

        if comm.rank == MpiReporter.MAIN:
            if log_folder is None:
                log_folder = datetime.now().strftime('es__%d_%m_%y__%H_%M_%S')

            if not path.exists(path.join('saved', log_folder)): os.makedirs(path.join('saved', log_folder))

            logging.basicConfig(filename=path.join('saved', log_folder, 'es.log'), level=logging.DEBUG)
            logging.info('initialized logger')

    def _print(self, s: str):
        logging.info(s)

    def _log(self, d: Dict[str, float]):
        for k, v in d.items():
            logging.info(f'{k}:{v}')


class MLFlowReporter(DefaultMpiReporter):
    def __init__(self, comm: MPI.Comm, cfg: Munch):
        super().__init__(comm)
        if comm.rank == MpiReporter.MAIN:
            set_experiment(cfg.env.name)
            start_run(run_name=cfg.general.name)
            log_params(json_normalize(unmunchify(cfg)).to_dict(orient='records')[0])

            self.gens = [0] * cfg.general.n_policies

            self.run_ids = []
            self.active_run = None
            for i in range(cfg.general.n_policies):
                with start_run(run_name=f'{i}', nested=True) as run:
                    self.run_ids.append(run.info.run_id)

    def set_active_run(self, i: int):
        if self.comm.rank == MpiReporter.MAIN:
            self.active_run = i

    def start_active_run(self):
        assert self.active_run is not None, \
            'No nested run is currently active, but you are trying to log metrics. Must call set_active_run first'

        return start_run(run_id=self.run_ids[self.active_run], nested=True)

    def _start_gen(self):
        pass

    def _end_gen(self):
        self.gens[self.active_run] += 1
        self.active_run = None

    def _print(self, s: str):
        pass

    def _log(self, d: Dict[str, float]):
        with self.start_active_run():
            log_metrics(d, self.gens[self.active_run])
