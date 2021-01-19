from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from src.utils.novelty import novelty


class TrainingResult(ABC):
    """Stores the results of a single training run"""

    def __init__(self, rewards: List[float], positions: List[float], obs: np.ndarray, steps: int, *args, **kwargs):
        self.rewards: List[float] = rewards
        self.positions: List[float] = positions
        self.obs: np.ndarray = obs
        self.steps = steps

    @property
    def ob_sum_sq_cnt(self) -> Tuple[np.ndarray, np.ndarray, int]:
        cnt = len(self.obs) if np.any(self.obs) else 0
        return self.obs.sum(axis=0), np.square(self.obs).sum(axis=0), cnt

    @abstractmethod
    def get_result(self) -> List[float]:
        pass

    result: List[float] = property(lambda self: self.get_result())
    reward = property(lambda self: sum(self.rewards))
    behaviour = property(lambda self: self.positions[-3:-1])


class RewardResult(TrainingResult):
    def get_result(self) -> List[float]:
        return [self.reward]


class MeanRewardResult(TrainingResult):
    def get_result(self) -> List[float]:
        return [self.reward / self.steps]


class DistResult(TrainingResult):
    def get_result(self) -> List[float]:
        return [np.linalg.norm(self.positions[-3:-1])]


class XDistResult(DistResult):
    def get_result(self) -> List[float]:
        return [self.positions[-3]]


class NSResult(TrainingResult):
    def __init__(self, rewards: List[float], positions: List[float], obs: np.ndarray, steps: int, archive: np.ndarray,
                 k: int):
        super().__init__(rewards, positions, obs, steps)
        self.archive = archive
        self.k = k

    novelty = property(lambda self: novelty(np.array(self.behaviour), self.archive, self.k))

    def get_result(self) -> List[float]:
        return [self.novelty]


class NSRResult(NSResult):
    def get_result(self) -> List[float]:
        return [sum(self.rewards), self.novelty]
