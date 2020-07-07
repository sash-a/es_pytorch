from typing import List, Tuple

import numpy as np
import torch

from noisetable import NoiseTable

from mpi4py.MPI import Comm


def norm_linear_weight_init(in_size: int, out_size: int):
    std = np.sqrt(2 / (in_size + out_size))
    return np.random.normal(0, std ** 2, (in_size, out_size))


def uniform_linear_weight_init(in_size: int, out_size: int):
    r = np.sqrt(6 / (in_size + out_size))
    return np.random.uniform(-r, r, (in_size, out_size))


def uniform_linear_bias_init(size: int):
    r = 1 / np.sqrt(size)
    return np.random.uniform(-r, r, size)


# Should this rather take in an nn.Module and set the params through the state dict?
#  Would be able to support more architectures but might be a bit slower
# TODO: GeneralGenome?
class Genome:
    def __init__(self, comm: Comm, layer_sizes: List[Tuple[int, int]], std: float):
        """
        :param layer_sizes: (input, output) size of each layer in the network
        :param std: the standard deviation of the normal distribution
        """
        self.layer_sizes: List[Tuple[int, int]] = layer_sizes
        self.std: float = std

        self.params: np.ndarray = np.empty(sum((i * o + o for i, o in layer_sizes)))
        if comm.rank == 0:
            self.params: np.ndarray = np.array([])
            for in_size, out_size in layer_sizes:
                weights = np.ndarray.flatten(uniform_linear_weight_init(in_size, out_size))
                bias = uniform_linear_bias_init(out_size)

                self.params = np.append(self.params, weights)
                self.params = np.append(self.params, bias)

        comm.Bcast(self.params)

    def pheno(self, noise_table: NoiseTable, seed=None) -> Tuple[torch.nn.Module, int]:
        """
        :returns the phenotype of the genome and the index used for the noise table so the params can be reconstructed
        """
        noise, idx = Genome.find_noise(noise_table, len(self.params), seed)
        return Genome.make_pheno(self, noise), idx

    @staticmethod
    def find_noise(noise_table: NoiseTable, size: int, seed=None) -> Tuple[np.ndarray, int]:
        """:returns a random subset of the noise table and the start position of the subset"""
        idx = np.random.RandomState(seed).randint(0, len(noise_table) - size)
        return noise_table.get(idx, size), idx

    @staticmethod
    def make_pheno(genome, noise: np.ndarray) -> torch.nn.Module:
        """Fills the weights of pytorch network of shape self.layer_sizes with self.params"""
        assert len(noise) == len(genome.params)

        layers: List[torch.nn.Module] = []
        curr_noise_pos = 0
        for in_size, out_size in genome.layer_sizes:
            weight_noise_end_pos = in_size * out_size + curr_noise_pos
            weight_noise = genome.std * noise[curr_noise_pos:weight_noise_end_pos]
            bias_noise = genome.std * noise[weight_noise_end_pos: weight_noise_end_pos + out_size]
            weights = weight_noise + genome.params[curr_noise_pos:in_size * out_size + curr_noise_pos]
            bias = bias_noise + genome.params[weight_noise_end_pos: weight_noise_end_pos + out_size]

            layer = torch.nn.Linear(in_size, out_size)
            layer.weight = torch.nn.Parameter(torch.from_numpy(np.reshape(weights, (out_size, in_size))).float(), False)
            layer.bias = torch.nn.Parameter(torch.from_numpy(bias).float(), False)

            layers.append(layer)
            layers.append(torch.nn.Tanh())

            curr_noise_pos = weight_noise_end_pos + out_size

        return torch.nn.Sequential(*layers)
