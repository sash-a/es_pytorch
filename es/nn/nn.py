import gym
import numpy as np
from torch import nn, Tensor, clamp

ob_clip = 5


class FullyConnected(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: int, n_hidden: int, activation: nn.Module,
                 env: gym.Env, policy_cfg):
        super().__init__()

        layers = [nn.Linear(in_size, hidden_size), activation()]
        for _ in range(n_hidden):
            layers += [nn.Linear(hidden_size, hidden_size), activation()]
        layers += [nn.Linear(hidden_size, out_size), activation()]

        self.model = nn.Sequential(*layers)
        self._action_std = policy_cfg.std

        self._obmean: np.ndarray = np.zeros(env.observation_space.shape)
        self._obstd: np.ndarray = np.ones(env.observation_space.shape)

    def forward(self, inp: Tensor, **kwargs):
        rs = kwargs['rs']

        inp = clamp((inp - self._obmean) / self._obstd, min=-ob_clip, max=ob_clip)
        a = self.model(inp.float())
        if self._action_std != 0 and rs is not None:
            a += rs.randn(*a.shape) * self._action_std

        return a.numpy()

    def set_ob_mean_std(self, mean: np.ndarray, std: np.ndarray):
        self._obmean = mean
        self._obstd = std
