from __future__ import annotations

from typing import Tuple

import numpy as np
from mpi4py import MPI


def create_shared_arr(comm: MPI.Comm, size: int) -> np.ndarray:
    itemsize: int = MPI.DOUBLE.Get_size()
    if comm.rank == 0:
        nbytes: int = size * itemsize
    else:
        nbytes: int = 0

    # Creating the shared block
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)

    # create a numpy array whose data points to the shared mem
    buf, itemsize = win.Shared_query(0)
    return np.ndarray(buffer=buf, dtype='d', shape=(size,))


class NoiseTable:
    def __init__(self, n_params: int, noise: np.ndarray):
        self.n_params = n_params
        self.noise = noise

    def get(self, i, size) -> np.ndarray:
        return self.noise[i:i + size]

    def sample(self, size=None, seed=None) -> Tuple[int, np.ndarray]:
        if size is None:
            size = self.n_params

        idx = np.random.RandomState(seed).randint(0, len(self) - size)
        return idx, self.get(idx, size)

    def __getitem__(self, item) -> np.ndarray:
        return self.get(item, self.n_params)

    def __len__(self):
        return len(self.noise)

    def __call__(self, *args, **kwargs) -> Tuple[int, np.ndarray]:
        return self.sample()

    @staticmethod
    def make_noise(size: int, seed=None) -> np.ndarray:
        return np.random.RandomState(seed).randn(size)

    @staticmethod
    def create_shared_noisetable(global_comm: MPI.Comm,
                                 size: int,
                                 n_params: int,
                                 seed=None) -> NoiseTable:

        local_comm: MPI.Comm = global_comm.Split_type(MPI.COMM_TYPE_SHARED)
        n_nodes = global_comm.allreduce(1 if local_comm.rank == 0 else 0, MPI.SUM)
        shared_arr = create_shared_arr(local_comm, size)
        nt = NoiseTable(n_params, shared_arr)

        if global_comm.rank == 0:
            noise = NoiseTable.make_noise(size, seed)  # create arr values

            for i in range(n_nodes - 1):
                global_rank_to_send = global_comm.recv(source=MPI.ANY_SOURCE)  # recv global rank from each nodes 0 proc
                print(f'Sending noise to rank: {global_rank_to_send}')
                global_comm.Send([noise + i + 1, MPI.DOUBLE], global_rank_to_send)  # send arr to that rank

            shared_arr[:size] = noise

        elif local_comm.rank == 0:
            noise = np.empty(size, dtype='d')
            global_comm.send(global_comm.rank, 0)  # send local rank
            global_comm.Recv([noise, MPI.DOUBLE], 0)  # receive arr values
            shared_arr[:size] = noise  # set array values

        global_comm.Barrier()  # wait until all nodes have set the array values
        return nt
