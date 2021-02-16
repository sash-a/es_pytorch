from __future__ import annotations

import os
import pickle

import numpy as np
import torch

from src.nn.nn import BaseNet
from src.nn.obstat import ObStat


def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)


class Policy(torch.nn.Module):
    def __init__(self, module: BaseNet, std: float):
        super().__init__()
        module.apply(init_normal)

        self._module: BaseNet = module
        self.std = std

        self.flat_params: np.ndarray = Policy.get_flat(module)
        self.obstat: ObStat = ObStat(module._obmean.shape, 1e-2)

    def __len__(self):
        return len(self.flat_params)

    @staticmethod
    def get_flat(module: torch.nn.Module) -> np.ndarray:
        return torch.cat([t.flatten() for t in module.state_dict().values()]).numpy()

    @staticmethod
    def load(file: str) -> Policy:
        policy: Policy = pickle.load(open(file, 'rb'))
        policy.set_nn_params(policy.flat_params)
        return policy

    def save(self, folder: str, suffix: str):
        if not os.path.exists(folder):
            os.makedirs(folder)

        pickle.dump(self, open(os.path.join(folder, f'policy-{suffix}'), 'wb'))

    def set_nn_params(self, params: np.ndarray) -> torch.nn.Module:
        with torch.no_grad():
            d = {}  # new state dict
            curr_param_idx = 0
            for name, weights in self._module.state_dict().items():
                n_params = weights.numel()
                d[name] = torch.from_numpy(np.reshape(params[curr_param_idx:curr_param_idx + n_params], weights.shape))
                curr_param_idx += n_params

            self._module.load_state_dict(d)
        return self._module

    def pheno(self, noise: np.ndarray = None) -> torch.nn.Module:
        if noise is None:
            noise = np.zeros(len(self))
        params = self.flat_params + self.std * noise
        self.set_nn_params(params)

        return self._module

    def update_obstat(self, obstat: ObStat):
        self.obstat += obstat  # adding the new observations to the global obstat
        self._module.set_ob_mean_std(self.obstat.mean, self.obstat.std)

    def forward(self, inp):
        self._module.forward(inp)
