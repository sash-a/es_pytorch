from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from utils.novelty import novelty


class TrainingResult(ABC):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float], *args, **kwargs):
        self.rewards: Sequence[float] = rewards
        self.behaviour: Sequence[float] = behaviour

    @abstractmethod
    def get_result(self) -> Sequence[float]:
        pass

    result: Sequence[float] = property(lambda self: self.get_result())


class RewardResult(TrainingResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float]):
        super().__init__(rewards, behaviour)

    def get_result(self):
        return [sum(self.rewards)]


class DistResult(TrainingResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float]):
        super().__init__(rewards, behaviour)

    def get_result(self):
        return [np.linalg.norm(self.behaviour[-3:-1])]


class NSResult(TrainingResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float], archive: np.ndarray, k: int):
        super().__init__(rewards, behaviour)
        self.archive = archive
        self.k = k

    def get_result(self):
        return [novelty(np.array([self.behaviour]), self.archive, self.k)]


class NSRResult(NSResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float], archive: np.ndarray, k: int):
        super().__init__(rewards, behaviour, archive, k)

    def get_result(self):
        return [sum(self.rewards), super().get_result()[0]]
