from typing import Tuple

import torch
import numpy as np

from noisetable import NoiseTable


class Policy(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, stdev: int):
        super().__init__()
        self._module: torch.nn.Module = module
        self.stdev = stdev

        self.flat_params: np.ndarray = Policy.get_flat(module)

    def __len__(self):
        return len(self.flat_params)

    @staticmethod
    def get_flat(module: torch.nn.Module) -> np.ndarray:
        return torch.cat([t.flatten() for t in module.state_dict().values()]).numpy()

    def set_nn_params(self, params: np.ndarray) -> torch.nn.Module:
        d = {}  # new state dict
        curr_params_idx = 0
        for name, weights in self._module.state_dict().items():
            n_params = torch.prod(torch.tensor(weights.shape))
            d[name] = torch.from_numpy(np.reshape(params[curr_params_idx:curr_params_idx + n_params], weights.size()))
            curr_params_idx += n_params

        self._module.load_state_dict(d)
        return self._module

    def pheno(self, noise: np.ndarray, sign: int) -> torch.nn.Module:
        assert sign == 1 or sign == -1

        params = self.flat_params + (sign * self.stdev) * noise
        self.set_nn_params(params)

        return self._module

    def forward(self, inp):
        self._module.forward(inp)
