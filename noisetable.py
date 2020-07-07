from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from mpi4py import MPI


class NoiseTable:
    def __init__(self, comm: MPI.Comm, seed=123, table_size=250000000):
        if comm.rank == 0:
            self.noise = np.random.RandomState(seed).randn(table_size)  # 64-bit to 32-bit conversion here
        else:
            self.noise: np.ndarray = np.empty(table_size)

        comm.Bcast(self.noise)

    def get(self, i, size) -> np.ndarray:
        return self.noise[i:i + size]

    def sample_index(self, stream, size) -> np.ndarray:
        return stream.randint(0, len(self.noise) - size + 1)

    def __len__(self):
        return len(self.noise)
