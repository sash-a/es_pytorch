from abc import ABC
from typing import List

import gym
import numpy as np
from torch import nn, Tensor, clamp


class BaseNet(nn.Module, ABC):
    def __init__(self, layers: List[nn.Module], ob_shape: tuple, ob_clip: float = 5):
        super().__init__()
        self.model = nn.Sequential(*layers)

        self._obmean: np.ndarray = np.zeros(ob_shape)
        self._obstd: np.ndarray = np.ones(ob_shape)

        self.ob_clip = ob_clip

    def set_ob_mean_std(self, mean: np.ndarray, std: np.ndarray):
        self._obmean = mean
        self._obstd = std


class FeedForward(BaseNet):
    def __init__(self, layer_sizes: List[int], activation: nn.Module, env: gym.Env, ac_std: float, ob_clip: float = 5):
        """
        Creates a basic feed forward network

        :param layer_sizes: the sizes of the hidden layers, input and output sizes are determined from the env
        :param ac_std: standard deviation of the gaussian actions
        :param ob_clip: min/max observation value
        """
        layer_sizes = [int(np.prod(env.observation_space.shape))] + layer_sizes + [int(np.prod(env.action_space.shape))]
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [nn.Linear(in_size, out_size), activation]

        super().__init__(layers, env.observation_space.shape, ob_clip)

        self._action_std = ac_std

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        rs = kwargs['rs']

        inp = clamp((inp - self._obmean) / self._obstd, min=-self.ob_clip, max=self.ob_clip)
        a = self.model(inp.float())
        if self._action_std != 0 and rs is not None:
            a += rs.randn(*a.shape) * self._action_std

        return a


class FFIntegGausAction(FeedForward):
    """
    Fully connected nn with integrated gaussian actions. Assumes that the first output of the network is the std for
    all the outputs. Assumes that the output is a 1D vector. Below is how the nn operates:

    out = model(input)
    act, std = out[1:], out[0]
    act += rs.standard_normal(*act.shape) * std
    """

    def forward(self, inp: Tensor, **kwargs) -> np.ndarray:
        rs: np.random.RandomState = kwargs['rs'] if 'rs' in kwargs else None

        inp = clamp((inp - self._obmean) / self._obstd, min=-self.ob_clip, max=self.ob_clip)
        out = self.model(inp.float()).numpy()

        # this class assumes that the first nn output is the std for all gaussian actions
        action, action_std = out[1:], out[0]
        if action_std != 0 and rs is not None:
            action += rs.standard_normal(*action.shape) * action_std

        return action


class FFIntegGausActionMulti(FeedForward):
    """
    Fully connected nn with integrated gaussian actions. Assumes that the first half of the output is the mean and the
    second half the std of the gauss actions
    """

    def forward(self, inp: Tensor, **kwargs) -> np.ndarray:
        rs: np.random.RandomState = kwargs['rs'] if 'rs' in kwargs else None

        inp = clamp((inp - self._obmean) / self._obstd, min=-self.ob_clip, max=self.ob_clip)
        out = self.model(inp.float()).numpy()

        # this class assumes that the first half of the nn output is the action and second half is the std
        mid = len(out) // 2
        action, action_std = out[:mid], np.abs(out[mid:])

        if rs is not None:
            action += rs.standard_normal(*action.shape) * action_std

        return action


class FFBinned(BaseNet):
    def __init__(self, layer_sizes: List[int], activation: nn.Module, env: gym.Env, n_bins: int, ob_clip=5):
        self.bins = n_bins
        self.adim, self.ahigh, self.alow = env.action_space.shape[0], env.action_space.high, env.action_space.low

        layer_sizes = [int(np.prod(env.observation_space.shape))] + layer_sizes + [self.adim * self.bins]
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [nn.Linear(in_size, out_size), activation]

        super().__init__(layers, env.observation_space.shape, ob_clip)

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        inp = clamp((inp - self._obmean) / self._obstd, min=-self.ob_clip, max=self.ob_clip)
        a: Tensor = self.model(inp.float())
        ac_range = (self.ahigh - self.alow)[None, :]

        binned_ac = a.reshape((-1, self.adim, self.bins)).argmax(2)
        return (1. / (self.bins - 1.) * binned_ac * ac_range + self.alow[None, :]).squeeze()
