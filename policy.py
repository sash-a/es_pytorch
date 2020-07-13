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
        self.n_params = len(self.flat_params)

    def __len__(self):
        return self.n_params

    @staticmethod
    def get_flat(module: torch.nn.Module) -> np.ndarray:
        return torch.cat([t.flatten() for t in module.state_dict().values()]).numpy()

    @staticmethod
    def set_module_params(module: torch.nn.Module, params: np.ndarray) -> torch.nn.Module:
        new_state_dict = {}
        curr_params_idx = 0
        for name, weights in module.state_dict().items():
            n_params = torch.prod(torch.tensor(weights.shape))
            new_state_dict[name] = \
                torch.from_numpy(np.reshape(params[curr_params_idx:curr_params_idx + n_params], weights.size()))
            curr_params_idx += n_params

        module.load_state_dict(new_state_dict)
        return module

    def pheno(self, nt: NoiseTable, seed=None) -> Tuple[torch.nn.Module, int]:
        idx = np.random.RandomState(seed).randint(0, len(nt) - self.n_params)
        noise = nt[idx]

        params = self.flat_params + self.stdev * noise
        Policy.set_module_params(self._module, params)

        return self._module, idx

    def forward(self, inp):
        self._module.forward(inp)
