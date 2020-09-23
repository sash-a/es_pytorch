import numpy as np

from es.evo.noisetable import create_shared_arr, NoiseTable
# noinspection PyUnresolvedReferences
from test import comm


def test_create_shared_arr(comm):
    size = 10
    shared_arr = create_shared_arr(comm, size)

    if comm.rank == 0:
        shared_arr[:size] = np.arange(size, dtype=np.float32)

    comm.barrier()
    assert (shared_arr == np.arange(size, dtype=np.float32)).all()


def test_create_shared(comm):
    size = 5
    seed = 1
    noise = np.array(NoiseTable.create_shared(comm, size, 0, seed=seed).noise)

    all_noise = comm.alltoall([noise] * comm.size)
    assert np.isclose(all_noise, all_noise[0]).all()
    assert np.isclose(all_noise[0], np.random.RandomState(seed).randn(size).astype(np.float32)).all()
