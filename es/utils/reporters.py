from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple

import numpy as np
from mlflow import log_params, log_metric, set_experiment, start_run
from mpi4py import MPI
from pandas import json_normalize

from es.evo.policy import Policy
from es.utils.TrainingResult import TrainingResult


def calc_dist_rew(tr: TrainingResult) -> Tuple[float, float]:
    # Calculating distance traveled (ignoring height dim). Assumes starting at 0, 0
    return np.linalg.norm(np.array(tr.behaviour[-3:-1])), np.sum(tr.rewards)


class Reporter(ABC):
    @abstractmethod
    def start_gen(self):
        pass

    @abstractmethod
    def end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                time: float):
        pass

    @abstractmethod
    def _print_fn(self, s: str):
        pass

    def print(self, s: str):
        self._print_fn(s)


class ReporterSet(Reporter):
    def __init__(self, *reporters: Reporter):
        self.reporters = reporters

    def start_gen(self):
        for reporter in self.reporters:
            reporter.start_gen()

    def end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                time: float):
        for reporter in self.reporters:
            reporter.end_gen(fits, noiseless_tr, noiseless_policy, steps, time)

    def _print_fn(self, s: str):
        raise NotImplementedError('Reporter set does not have a print_fn it can only print')

    def print(self, s: str):
        for reporter in self.reporters:
            reporter.print(s)


class MPIReporter(Reporter, ABC):
    ROOT = 0

    def __init__(self, comm: MPI.Comm):
        self.comm = comm

    def start_gen(self):
        if self.comm.rank == MPIReporter.ROOT:
            self._start_gen()

    def end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                time: float):
        if self.comm.rank == MPIReporter.ROOT:
            self._end_gen(fits, noiseless_tr, noiseless_policy, steps, time)

    @abstractmethod
    def _start_gen(self):
        pass

    @abstractmethod
    def _end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                 time: float):
        pass

    def print(self, s: str):
        if self.comm.rank == MPIReporter.ROOT:
            super().print(s)


class StdoutReporter(MPIReporter):
    def __init__(self, comm: MPI.Comm):
        super().__init__(comm)
        if comm.rank == 0:
            self.gen = 0
            self.cum_steps = 0

    def _start_gen(self):
        print(f'\n\n'
              f'----------------------------------------'
              f'\ngen:{self.gen}')

    def _end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                 time: float):
        for i, col in enumerate(fits.T):
            # Objectives are grouped by column so this finds the avg and max of each objective
            print(f'obj {i} avg:{np.mean(col):0.2f}')
            print(f'obj {i} max:{np.max(col):0.2f}')

        print(f'fit:{noiseless_tr.result}')
        dist, rew = calc_dist_rew(noiseless_tr)
        self.cum_steps += steps

        print(f'dist:{dist}')
        print(f'rew:{rew}')

        print(f'steps:{steps}')
        print(f'cum steps:{self.cum_steps}')
        print(f'time:{time:0.2f}')
        self.gen += 1

    def _print_fn(self, s: str):
        print(s)


class LoggerReporter(MPIReporter):
    def __init__(self, comm: MPI.Comm, cfg, log_name=None):
        super().__init__(comm)

        if log_name is None:
            log_name = datetime.now().strftime('es__%d_%m_%y__%H_%M_%S')
        logging.basicConfig(filename=f'logs/{log_name}.log', level=logging.DEBUG)
        logging.info('initialized logger')

        if comm.rank == 0:
            self.gen = 0
            self.cfg = cfg

            self.best_rew = 0
            self.best_dist = 0
            self.cum_steps = 0

    def _start_gen(self):
        logging.info(f'gen:{self.gen}')

    def _end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                 time: float):
        for i, col in enumerate(fits.T):
            # Objectives are grouped by column so this finds the avg and max of each objective
            logging.info(f'obj {i} avg:{np.mean(col):0.2f}')
            logging.info(f'obj {i} max:{np.max(col):0.2f}')

        logging.info(f'fit:{noiseless_tr.result}')
        dist, rew = calc_dist_rew(noiseless_tr)
        self.cum_steps += steps

        logging.info(f'dist:{dist}')
        logging.info(f'rew:{rew}')

        logging.info(f'steps:{steps}')
        logging.info(f'cum steps:{self.cum_steps}')
        logging.info(f'time:{time:0.2f}')
        self.gen += 1

    def _print_fn(self, s: str):
        logging.info(s)


class MLFlowReporter(MPIReporter):
    def __init__(self, comm: MPI.Comm, cfg_file: str, cfg):
        super().__init__(comm)

        if comm.rank == 0:
            set_experiment(cfg.env.name)
            start_run(run_name=cfg.general.name)
            log_params(json_normalize(json.load(open(cfg_file))).to_dict(orient='records')[0])

            self.gen = 0
            self.best_rew = 0
            self.best_dist = 0
            self.cum_steps = 0

    def _start_gen(self):
        pass

    def _end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                 time: float):
        for i, col in enumerate(fits.T):
            # Objectives are grouped by column so this finds the avg and max of each objective
            log_metric(f'obj {i} avg', np.mean(col), self.gen)
            log_metric(f'obj {i} max', np.max(col), self.gen)

        dist, rew = calc_dist_rew(noiseless_tr)
        self.cum_steps += steps

        log_metric('dist', dist, self.gen)
        log_metric('rew', rew, self.gen)
        log_metric(f'steps', steps, self.gen)
        log_metric(f'cum steps', self.cum_steps, self.gen)
        log_metric('time', time, self.gen)

        self.gen += 1

    def _print_fn(self, s: str):
        # key_val = s.split(':')
        # if len(key_val) == 2:
        #     log_param(key_val[0], key_val[1])
        pass
