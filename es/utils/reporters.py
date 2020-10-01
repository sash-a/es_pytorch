from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
from mlflow import log_params, log_metric, log_metrics, set_experiment, start_run
from mpi4py import MPI
from pandas import json_normalize

from es.evo.policy import Policy
from es.utils.training_result import TrainingResult


def calc_dist_rew(tr: TrainingResult) -> Tuple[float, float]:
    # Calculating distance traveled (ignoring height dim). Assumes starting at 0, 0
    return np.linalg.norm(np.array(tr.positions[-3:-1])), np.sum(tr.rewards)


class Reporter(ABC):
    @abstractmethod
    def start_gen(self):
        pass

    @abstractmethod
    def end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                time: float):
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

    def end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                time: float):
        for reporter in self.reporters:
            reporter.end_gen(fits, noiseless_tr, noiseless_policy, steps, time)

    def print(self, s: str):
        for reporter in self.reporters:
            reporter.print(s)

    def log(self, d: Dict[str, float]):
        for reporter in self.reporters:
            reporter.log(d)


class MPIReporter(Reporter, ABC):
    MAIN = 0

    def __init__(self, comm: MPI.Comm):
        self.comm = comm

    def start_gen(self):
        if self.comm.rank == MPIReporter.MAIN:
            self._start_gen()

    def end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                time: float):
        if self.comm.rank == MPIReporter.MAIN:
            self._end_gen(fits, noiseless_tr, noiseless_policy, steps, time)

    def print(self, s: str):
        if self.comm.rank == MPIReporter.MAIN:
            self._print(s)

    def log(self, d: Dict[str, float]):
        if self.comm.rank == MPIReporter.MAIN:
            self._log(d)

    @abstractmethod
    def _start_gen(self):
        pass

    @abstractmethod
    def _end_gen(self, fits: np.ndarray, noiseless_tr: TrainingResult, noiseless_policy: Policy, steps: int,
                 time: float):
        pass

    @abstractmethod
    def _print(self, s: str):
        pass

    @abstractmethod
    def _log(self, d: Dict[str, float]):
        pass


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

    def _print(self, s: str):
        print(s)

    def _log(self, d: Dict[str, float]):
        for k, v in d.items():
            print(f'{k}:{v}')


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

    def _print(self, s: str):
        logging.info(s)

    def _log(self, d: Dict[str, float]):
        for k, v in d.items():
            logging.info(f'{k}:{v}')


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

    def _print(self, s: str):
        pass

    def _log(self, d: Dict[str, float]):
        log_metrics(d, self.gen)
