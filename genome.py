from typing import List, Tuple

import numpy as np
import torch

from noisetable import NoiseTable


def norm_linear_weight_init(in_size: int, out_size: int):
    std = np.sqrt(2 / (in_size + out_size))
    return np.random.normal(0, std ** 2, (in_size, out_size))


def uniform_linear_weight_init(in_size: int, out_size: int):
    range = np.sqrt(6 / (in_size + out_size))
    return np.random.uniform(-range, range, (in_size, out_size))


class Genome:
    def __init__(self, layer_sizes: List[Tuple[int, int]], noise: NoiseTable):
        self.fitness: float = 0.
        self.noise_table: NoiseTable = noise
        self.layer_sizes: List[Tuple[int, int]] = layer_sizes

        self.params: np.ndarray = np.array([])
        for in_size, out_size in layer_sizes:
            weights = np.ndarray.flatten(uniform_linear_weight_init(in_size, out_size))
            self.params = np.append(self.params, weights)

    def to_pheno(self, seed):
        n_params = len(self.params)
        noise = self.noise_table.get(np.random.RandomState(seed).randint(0, len(self.noise_table) - n_params), n_params)

        layers: List[torch.nn.Module] = []
        curr_noise_pos = 0
        for in_size, out_size in self.layer_sizes:
            weights_noise = noise[curr_noise_pos:in_size * out_size + curr_noise_pos]
            weights = weights_noise + self.params[curr_noise_pos:in_size * out_size + curr_noise_pos]

            layer = torch.nn.Linear(in_size, out_size)
            layer.weight = torch.nn.Parameter(torch.from_numpy(np.reshape(weights, (out_size, in_size))).float(),
                                              False).float()
            # TODO bias

            layers.append(layer)

        return torch.nn.Sequential(*layers)
