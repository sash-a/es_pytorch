# Adapted from: https://github.com/uber-research/deep-neuroevolution
from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self, dim: int, lr: float):
        self.lr: float = lr
        self.dim: int = dim
        self.t: int = 0

    def step(self, globalg):
        """
        Returns the step to take the for the policies params

        :param globalg: the average of the sum of the noises weighted by the rewards they received
        :return: the step to take the for the policies params
        """
        self.t += 1
        return self._compute_step(globalg)

    @abstractmethod
    def _compute_step(self, globalg):
        pass


class SimpleES(Optimizer):
    def __init__(self, dim: int, lr: float):
        super().__init__(dim, lr)

    def _compute_step(self, globalg: np.ndarray):
        return self.lr * globalg


class SGD(Optimizer):
    def __init__(self, dim: int, lr: float, momentum=0.9):
        Optimizer.__init__(self, dim, lr)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.momentum = momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        return -self.lr * self.v


class Adam(Optimizer):
    def __init__(self, dim: int, lr: float, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, dim, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalgrad):
        a = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalgrad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalgrad * globalgrad)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
