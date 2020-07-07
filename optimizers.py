# https://github.com/uber-research/deep-neuroevolution

import numpy as np


class Optimizer(object):
    def __init__(self, theta):
        self.theta = theta
        self.dim = len(self.theta)
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.theta
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        new_theta = self.theta + step
        self.theta = new_theta
        return ratio, new_theta

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, theta, stepsize, momentum=0.9):
        Optimizer.__init__(self, theta)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, theta, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, theta)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalgrad):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalgrad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalgrad * globalgrad)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
