from __future__ import annotations

import numpy as np


class Reporter:
    def report_fits(self, gen: int, fits: np.ndarray):
        avg = np.mean(fits)
        mx = np.max(fits)
        print(f'Gen:{gen}-avg:{avg:0.2f}-max:{mx:0.2f}')

    def report_noiseless(self, gen: int, fit: float):
        """Reports the fitness of a evaluation using no noise from the table and noiseless actions"""
        print(f'Gen:{gen}-noiseless:{fit}')
