# Adapted from: https://github.com/uber-research/deep-neuroevolution
from abc import ABC, abstractmethod

import numpy as np

from es.evo.policy import Policy


class Optimizer(ABC):
    def __init__(self, policy: Policy, lr: float):
        self.policy: Policy = policy
        self.lr: float = lr
        self.dim: int = len(policy)
        self.t: int = 0

    def step(self, globalg):
        """
        Updates the flat_params of the policy
        :param globalg: the average of the sum of the noises weighted by the rewards they received
        :return: the new flat_params
        """
        self.t += 1
        self.policy.flat_params += self._compute_step(globalg)
        return self.policy.flat_params

    @abstractmethod
    def _compute_step(self, globalg):
        pass


class ES(Optimizer):
    def __init__(self, policy: Policy, lr: float):
        super().__init__(policy, lr)

    def _compute_step(self, globalg: np.ndarray):
        return (self.lr / self.policy.std) * globalg


class SGD(Optimizer):
    def __init__(self, policy: Policy, lr, momentum=0.9):
        Optimizer.__init__(self, policy, lr)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.momentum = momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        return -self.lr * self.v


class Adam(Optimizer):
    def __init__(self, policy: Policy, lr, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, policy, lr)
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
