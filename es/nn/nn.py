import gym
import numpy as np
from torch import nn, Tensor, clamp

ob_clip = 5


class FullyConnected(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: int, n_hidden: int, activation: nn.Module,
                 env: gym.Env, policy_cfg):
        super().__init__()

        layers = [nn.Linear(in_size, hidden_size), activation]
        for _ in range(n_hidden):
            layers += [nn.Linear(hidden_size, hidden_size), activation]
        layers += [nn.Linear(hidden_size, out_size), activation]

        self.model = nn.Sequential(*layers)
        self._action_std = policy_cfg.ac_std

        self._obmean: np.ndarray = np.zeros(env.observation_space.shape)
        self._obstd: np.ndarray = np.ones(env.observation_space.shape)

    def forward(self, inp: Tensor, **kwargs):
        rs = kwargs['rs'] if 'rs' in kwargs else None
        to_int = kwargs['to_int'] if 'to_int' in kwargs else None

        inp = clamp((inp - self._obmean) / self._obstd, min=-ob_clip, max=ob_clip)
        a: Tensor = self.model(inp.float())
        if self._action_std != 0 and rs and not to_int:
            a += rs.randn(*a.shape) * self._action_std

        if to_int and not rs:  # round to -1 if < -0.5 to to 1 if > 0.5 and 0 otherwise
            a[a > 0.5] = 1
            a[a < -0.5] = -1
            btw_idx = np.where((a > -0.5) & (a < 0.5))
            a[btw_idx] = 0

        return a.numpy().astype(int)

    def set_ob_mean_std(self, mean: np.ndarray, std: np.ndarray):
        self._obmean = mean
        self._obstd = std
