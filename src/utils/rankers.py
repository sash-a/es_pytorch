from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


def rank(x: np.ndarray):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


class Ranker(ABC):
    """Ranks all fitnesses obtained in a generation"""

    def __init__(self):
        self.fits_pos: Optional[np.ndarray] = None
        self.fits_neg: Optional[np.ndarray] = None
        self.noise_inds: Optional[np.ndarray] = None
        self.ranked_fits: Optional[np.ndarray] = None
        self.n_fits_ranked: int = 0

    fits = property(lambda self: np.concatenate((self.fits_pos, self.fits_neg)))

    @abstractmethod
    def _rank(self, x: np.ndarray) -> np.ndarray:
        """Ranks self.fits"""
        pass

    def _pre_rank(self, fits_pos: np.ndarray, fits_neg: np.ndarray, noise_inds: np.ndarray):
        self.fits_pos = fits_pos
        self.fits_neg = fits_neg
        self.noise_inds = noise_inds

    def _post_rank(self, ranked_fits: np.ndarray) -> np.ndarray:
        self.n_fits_ranked = ranked_fits.size
        return ranked_fits[:len(self.fits_pos)] - ranked_fits[len(self.fits_pos):]

    def rank(self, fits_pos: np.ndarray, fits_neg: np.ndarray, noise_inds: np.ndarray) -> np.ndarray:
        self._pre_rank(fits_pos, fits_neg, noise_inds)
        ranked_fits = self._rank(self.fits)
        self.ranked_fits = self._post_rank(ranked_fits)
        return self.ranked_fits


class CenteredRanker(Ranker):
    def _rank(self, x: np.ndarray) -> np.ndarray:
        y = rank(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= (x.size - 1)
        y -= .5
        return np.squeeze(y)


class DoublePositiveCenteredRanker(CenteredRanker):
    def _rank(self, x: np.ndarray) -> np.ndarray:
        y = super()._rank(x)
        y[y > 0] *= 2
        return y


class MaxNormalizedRanker(Ranker):
    def _rank(self, x: np.ndarray) -> np.ndarray:
        mn = np.min(x)
        y = x + (-mn if mn > 0 else mn)  # brings minimum value to 0 -> [0, max(x)]
        y /= np.max(y)  # normalizes fits -> [0, 1]
        y = 2 * y - 1  # stretch fits out -> [-1, 1]
        return np.squeeze(y)


class SemiCenteredRanker(Ranker):
    def _rank(self, x: np.ndarray) -> np.ndarray:
        y = rank(x.ravel()).reshape(x.shape).astype(np.float32)
        s = x.size
        y = (((1 / s) * np.square(y + 0.29 * s)) / s) - 0.5
        return y


class EliteRanker(Ranker):
    def __init__(self, ranker: Ranker, elite_percent: float):
        super().__init__()
        assert 0 <= elite_percent <= 1
        self.ranker = ranker
        self.elite_percent = elite_percent

    def _rank(self, x: np.ndarray) -> np.ndarray:
        ranked = self.ranker._rank(self.fits)
        n_elite = max(1, int(ranked.size * self.elite_percent))
        elite_fit_inds = np.argpartition(ranked, -n_elite)[-n_elite:]
        # setting the noise inds to only be the inds of the elite
        self.noise_inds = self.noise_inds[elite_fit_inds % len(self.noise_inds)]
        return ranked[elite_fit_inds]

    # This needs to be redefined here as don't want to subtract any fits
    def _post_rank(self, ranked_fits: np.ndarray) -> np.ndarray:
        self.n_fits_ranked = ranked_fits.size
        return ranked_fits


class MultiObjectiveRanker(Ranker):
    def __init__(self, ranker: Ranker, w: float):
        assert 0. <= w <= 1.
        super().__init__()
        self.ranker = ranker
        self.w = w

    def _rank(self, x: np.ndarray) -> np.ndarray:
        assert x.shape[1] == 2  # this only works for 2 objectives

        ranked = []
        for objective_fits in x.T:
            ranked.append(self.ranker._rank(objective_fits))

        return ranked[0] * self.w + ranked[1] * (1 - self.w)
