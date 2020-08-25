from abc import ABC, abstractmethod
from typing import Sequence, List

import numpy as np

from es.utils.novelty import novelty


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

    def get_result(self) -> List[float]:
        return [sum(self.rewards)]


class DistResult(TrainingResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float]):
        super().__init__(rewards, behaviour)

    def get_result(self) -> List[float]:
        return [np.linalg.norm(self.behaviour[-3:-1])]


class XDistResult(DistResult):
    def get_result(self) -> List[float]:
        return [self.behaviour[-3]]


class NSResult(TrainingResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float], archive: np.ndarray, k: int):
        super().__init__(rewards, behaviour)
        self.archive = archive
        self.k = k

    def get_result(self) -> List[float]:
        return [novelty(np.array(self.behaviour[-3:]), self.archive, self.k)]


class NSRResult(NSResult):
    def __init__(self, rewards: Sequence[float], behaviour: Sequence[float], archive: np.ndarray, k: int):
        super().__init__(rewards, behaviour, archive, k)

    def get_result(self) -> List[float]:
        return [sum(self.rewards), super().get_result()[0]]
