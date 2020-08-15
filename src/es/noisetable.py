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

    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)  # Creating the shared block

    # create a numpy array whose data points to the shared mem
    buf, itemsize = win.Shared_query(0)
    return np.ndarray(buffer=buf, dtype='d', shape=(size,))


class NoiseTable:
    def __init__(self, n_params: int, noise: np.ndarray):
        self.n_params: int = n_params
        self.noise: np.ndarray = noise

    def get(self, i, size) -> np.ndarray:
        return self.noise[i:i + size]

    def sample_idx(self, rs: np.random.RandomState, size: int):
        return rs.randint(0, len(self) - size)

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
        """Shares a noise table across multiple nodes. Assumes that each node has at least 2 MPI processes"""
        local_comm: MPI.Comm = global_comm.Split_type(MPI.COMM_TYPE_SHARED)

        n_nodes = global_comm.allreduce(1 if local_comm.rank == 0 else 0, MPI.SUM)

        shared_arr = create_shared_arr(local_comm, size)
        nt = NoiseTable(n_params, shared_arr)

        if global_comm.rank == 0:
            # create and distribute seed
            seed = seed if seed is not None else np.random.randint(0, 10000)  # create seed if one is not provided
            for i in range(n_nodes):
                global_rank_to_send = global_comm.recv(source=MPI.ANY_SOURCE)  # recv global rank from each nodes 0 proc
                print(f'Sending seed {seed} to rank {global_rank_to_send}')
                global_comm.send(seed, global_rank_to_send)  # send seed to that rank

        if local_comm.rank == 1:
            # send rank, receive seed and populated shared mem with noise
            global_comm.send(global_comm.rank, 0)  # send local rank
            seed = global_comm.recv(source=0)  # receive noise seed
            print(f'Rank {global_comm.rank} received seed {seed}')

            noise = NoiseTable.make_noise(size, seed)  # create arr values
            shared_arr[:size] = noise  # set array values

        global_comm.Barrier()  # wait until all nodes have set the array values
        check_values(global_comm, nt)
        return nt


def check_values(comm: MPI.Comm, nt: NoiseTable):
    my_sample = np.concatenate((nt.noise[:5], nt.noise[-5:]))
    other_sample = np.empty(my_sample.shape)

    if comm.rank == 0:
        comm.Send([my_sample, MPI.DOUBLE], (comm.rank + 1) % comm.size)

    comm.Recv(other_sample, (comm.rank - 1) % comm.size)

    if comm.rank != 0:
        comm.Send([my_sample, MPI.DOUBLE], (comm.rank + 1) % comm.size)

    if not (my_sample == other_sample).all():
        print(f'rank {comm.rank} sample: {my_sample} != {other_sample}')
        raise Exception(f'rank {comm.rank} sample: {my_sample} != {other_sample}')
    else:
        print(f'rank {comm.rank} has same noise values as {(comm.rank - 1) % comm.size}')
