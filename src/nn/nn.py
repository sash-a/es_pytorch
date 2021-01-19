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


class FCIntegGausAction(FullyConnected):
    """
    Fully connected nn with integrated gaussian actions. Assumes that the first output of the network is the std for
    all the outputs. Assumes that the output is a 1D vector. Below is how the nn operates:

    out = model(input)
    act, std = out[1:], out[0]
    act += rs.standard_normal(*act.shape) * std
    """

    def forward(self, inp: Tensor, **kwargs) -> np.ndarray:
        rs: np.random.RandomState = kwargs['rs'] if 'rs' in kwargs else None

        inp = clamp((inp - self._obmean) / self._obstd, min=-ob_clip, max=ob_clip)
        out = self.model(inp.float()).numpy()

        # this class assumes that the first nn output is the std for all gaussian actions
        action, action_std = out[1:], out[0]
        if action_std != 0 and rs is not None:
            action += rs.standard_normal(*action.shape) * action_std

        return action


class FCIntegGausActionMulti(FullyConnected):
    """
    Fully connected nn with integrated gaussian actions. Assumes that the first half of the output is the mean and the
    second half the std of the gauss actions
    """

    def forward(self, inp: Tensor, **kwargs) -> np.ndarray:
        rs: np.random.RandomState = kwargs['rs'] if 'rs' in kwargs else None

        inp = clamp((inp - self._obmean) / self._obstd, min=-ob_clip, max=ob_clip)
        out = self.model(inp.float()).numpy()

        # this class assumes that the first half of the nn output is the action and second half is the std
        mid = len(out) // 2
        action, action_std = out[:mid], np.abs(out[mid:])

        if rs is not None:
            action += rs.standard_normal(*action.shape) * action_std

        return action
