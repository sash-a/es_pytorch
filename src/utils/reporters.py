from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np


class Reporter(ABC):
    @abstractmethod
    def report_fits(self, gen: int, fits: np.ndarray):
        pass

    @abstractmethod
    def report_noiseless(self, gen: int, fit: float):
        pass


class StdoutReporter(Reporter):
    def report_fits(self, gen: int, fits: np.ndarray):
        avg = np.mean(fits)
        mx = np.max(fits)
        print(f'Gen:{gen}-avg:{avg:0.2f}-max:{mx:0.2f}')

    def report_noiseless(self, gen: int, fit: float):
        """Reports the fitness of a evaluation using no noise from the table and noiseless actions"""
        print(f'Gen:{gen}-noiseless:{fit}')


class LoggerReporter(Reporter):
    def __init__(self, log_name=None):
        if log_name is None:
            log_name = datetime.now().strftime('es=%d-%m-%y_%H:%M:%S.log')
        logging.basicConfig(filename=log_name, level=logging.DEBUG)

    def report_fits(self, gen: int, fits: np.ndarray):
        avg = np.mean(fits)
        mx = np.max(fits)
        logging.info(f'Gen:{gen}-avg:{avg:0.2f}-max:{mx:0.2f}')

    def report_noiseless(self, gen: int, fit: float):
        logging.info(f'Gen:{gen}-noiseless:{fit}')
