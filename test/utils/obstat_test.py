import numpy as np

from es.utils.ObStat import ObStat
# noinspection PyUnresolvedReferences
from test import comm


def test_mpi_sum_obstat(comm):
    ob_size = 5
    obstat = ObStat(ob_size, 0)
    obstat.inc(np.arange(ob_size) * (comm.rank + 1), np.square(np.arange(ob_size) * (comm.rank + 1)), 1)

    obstat.mpi_inc(comm)

    expected_sum = np.zeros(ob_size)
    expected_square = np.zeros(ob_size)
    for i in range(comm.size):
        expected_sum += np.arange(ob_size) * (i + 1)
        expected_square += np.square(np.arange(ob_size) * (i + 1))

    assert (obstat.sum == expected_sum).all()
    assert (obstat.sumsq == expected_square).all()
    assert obstat.count == comm.size
