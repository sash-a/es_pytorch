from __future__ import annotations

from typing import Tuple, Optional

import gym
import numpy as np
from gym.utils.seeding import _int_list_from_bigint
from mpi4py import MPI

from src.utils.reporters import Reporter


def create_shared_arr(comm: MPI.Comm, size: int) -> np.ndarray:
    itemsize: int = MPI.FLOAT.Get_size()
    if comm.rank == 0:
        nbytes: int = size * itemsize
    else:
        nbytes: int = 0

    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)  # Creating the shared block

    # create a numpy array whose data points to the shared mem
    buf, itemsize = win.Shared_query(0)
    return np.ndarray(buffer=buf, dtype=np.float32, shape=(size,))


class NoiseTable:
    def __init__(self, n_params: int, noise: np.ndarray):
        self.n_params: int = n_params
        self.noise: np.ndarray = noise
        self._size = len(noise)

    def get(self, i, size) -> np.ndarray:
        assert len(self) > i + size, 'trying to index outside the range of the noise table'
        return self.noise[i:i + size]

    def sample_idx(self, rs: np.random.RandomState, size: int):
        upper_bound = len(self) - size
        if upper_bound <= 0: raise ValueError(f'Network (size:{size}) is too large for noise table (size:{len(self)})')
        return rs.randint(0, upper_bound)

    def sample(self, rs: np.random.RandomState = None, size=None) -> Tuple[int, np.ndarray]:
        if size is None:
            size = self.n_params

        if rs is None:
            rs = np.random.RandomState()

        idx = self.sample_idx(rs, size)
        return idx, self.get(idx, size)

    def __getitem__(self, item) -> np.ndarray:
        return self.get(item, self.n_params)

    def __len__(self):
        return self._size

    def __call__(self, *args, **kwargs) -> Tuple[int, np.ndarray]:
        return self.sample()

    @staticmethod
    def make_noise(size: int, seed=None) -> np.ndarray:
        rs, _ = gym.utils.seeding.np_random(seed)
        return rs.randn(size).astype(np.float32)

    @staticmethod
    def create_shared(global_comm: MPI.Comm, size: int, n_params: int, reporter: Optional[Reporter] = None,
                      seed=None) -> NoiseTable:
        """Shares a noise table across multiple nodes. Assumes that each node has at least 2 MPI processes"""
        local_comm: MPI.Comm = global_comm.Split_type(MPI.COMM_TYPE_SHARED)
        assert local_comm.size > 1

        n_nodes = global_comm.allreduce(1 if local_comm.rank == 0 else 0, MPI.SUM)

        shared_arr = create_shared_arr(local_comm, size)
        nt = NoiseTable(n_params, shared_arr)

        if global_comm.rank == 0:  # create and distribute seed
            seed = seed if seed is not None else np.random.randint(0, 1000000)  # create seed if one is not provided
            if reporter is not None: reporter.print(f'nt seed:{seed}')
            for i in range(n_nodes):
                global_rank_to_send = global_comm.recv(source=MPI.ANY_SOURCE)  # recv global rank from each nodes proc 1
                global_comm.send(seed, global_rank_to_send)  # send seed to that rank

        if local_comm.rank == 1:  # send rank, receive seed and populated shared mem with noise
            global_comm.send(global_comm.rank, 0)  # send local rank
            seed = global_comm.recv(source=0)  # receive noise seed
            shared_arr[:size] = NoiseTable.make_noise(size, seed)  # create arr values

        global_comm.Barrier()  # wait until all nodes have set the array values
        return nt
