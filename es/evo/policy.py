from __future__ import annotations

import os
import pickle

import numpy as np
import torch


def init_normal(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)


class Policy(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, std: float):
        super().__init__()
        module.apply(init_normal)

        self._module: torch.nn.Module = module
        self.std = std

        self.w = 0.5  # weighting of novelty in ranking function
        self.flat_params: np.ndarray = Policy.get_flat(module)
        # self.activated_w = torch.nn.functional.sigmoid(self.learned_w)

    flat_params_w = property(lambda self: np.concatenate((self.flat_params, np.array([self.w]))))

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
            curr_params_idx = 0
            for name, weights in self._module.state_dict().items():
                n_params = torch.prod(torch.tensor(weights.shape))
                d[name] = torch.from_numpy(
                    np.reshape(params[curr_params_idx:curr_params_idx + n_params], weights.size()))
                curr_params_idx += n_params

            self._module.load_state_dict(d)
        return self._module

    def pheno(self, noise: np.ndarray = None) -> torch.nn.Module:
        if noise is None:
            noise = np.zeros(len(self))
        params = self.flat_params + self.std * noise
        self.set_nn_params(params)

        return self._module

    def forward(self, inp):
        self._module.forward(inp)
