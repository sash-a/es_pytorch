from abc import ABC, abstractmethod
from typing import Sequence, List, Tuple

import numpy as np

from es.utils.novelty import novelty


class TrainingResult(ABC):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float], obs: np.ndarray, steps: int,
                 *args, **kwargs):
        self.rewards: Sequence[float] = rewards
        self.behaviour: Sequence[float] = behaviour
        self.obs: np.ndarray = obs
        self.steps = steps

    @property
    def ob_sum_sq_cnt(self) -> Tuple[np.ndarray, np.ndarray, int]:
        cnt = len(self.obs) if np.any(self.obs) else 0
        return self.obs.sum(axis=0), np.square(self.obs).sum(axis=0), cnt

    @abstractmethod
    def get_result(self) -> Sequence[float]:
        pass

    result: Sequence[float] = property(lambda self: self.get_result())


class RewardResult(TrainingResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float], obs: np.ndarray, steps: int):
        super().__init__(rewards, behaviour, obs, steps)

    def get_result(self) -> List[float]:
        return [sum(self.rewards)]


class DistResult(TrainingResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float], obs: np.ndarray, steps: int):
        super().__init__(rewards, behaviour, obs, steps)

    def get_result(self) -> List[float]:
        return [np.linalg.norm(self.behaviour[-3:-1])]


class XDistResult(DistResult):
    def get_result(self) -> List[float]:
        return [self.behaviour[-3]]


class NSResult(TrainingResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float], obs: np.ndarray, steps: int,
                 archive: np.ndarray, k: int):
        super().__init__(rewards, behaviour, obs, steps)
        self.archive = archive
        self.k = k

    def get_result(self) -> List[float]:
        return [novelty(np.array(self.behaviour[-3:-1]), self.archive, self.k)]


class NSRResult(NSResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float], obs: np.ndarray, steps: int,
                 archive: np.ndarray, k: int):
        super().__init__(rewards, behaviour, obs, steps, archive, k)

    def get_result(self) -> List[float]:
        return [self.behaviour[-3], super().get_result()[0]]
