from __future__ import annotations

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
    def __init__(self, comm: MPI.Comm, table_size: int, n_params: int, seed=None):
        self.n_params = n_params

        local_comm: MPI.Comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        self.noise: np.ndarray = create_shared_arr(local_comm, table_size)
        self._share(comm, local_comm, table_size, seed)

    def sample_index(self, stream, size) -> np.ndarray:
        return stream.randint(0, len(self.noise) - size + 1)

    def get(self, i, size) -> np.ndarray:
        return self.noise[i:i + size]

    def __getitem__(self, item) -> np.ndarray:
        return self.get(item, self.n_params)

    def __len__(self):
        return len(self.noise)

    def _share(self, global_comm: MPI.Comm, local_comm: MPI.Comm, size: int, seed):
        """Shares the noise table such that each node has 1 copy"""
        n_nodes = global_comm.allreduce(1 if local_comm.rank == 0 else 0, MPI.SUM)

        if global_comm.rank == 0:
            # create arr values
            noise = np.random.RandomState(seed).randn(size)

            for i in range(n_nodes - 1):
                global_rank_to_send = global_comm.recv(source=MPI.ANY_SOURCE)  # recv global rank from each nodes 0 proc
                print(f'Sending noise to rank: {global_rank_to_send}')
                global_comm.Send([noise + i + 1, MPI.DOUBLE], global_rank_to_send)  # send arr to that rank

            self.noise[:size] = noise

        elif local_comm.rank == 0:
            buf = np.empty(size, dtype='d')
            global_comm.send(global_comm.rank, 0)  # send local rank
            global_comm.Recv([buf, MPI.DOUBLE], 0)  # receive arr values
            self.noise[:size] = buf  # set array values

        global_comm.Barrier()  # wait until all nodes have set the array values
