# Very informal test to make sure that parallelism is working

import numpy as np
import pytest

from src.es.es_runner import _share_results


@pytest.fixture
def comm():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def test__share_results(comm):
    assert comm.size == 2
    if comm.rank == 0:
        inds = [10, 20, 30]
        fits_pos = [[1., 1.], [2., 2.], [3., 3.]]
        fits_neg = [[-1., -1.], [-2., -2.], [-3., -3.]]
    else:
        inds = [40, 50, 60]
        fits_pos = [[4., 4.], [5., 5.], [6., 6.]]
        fits_neg = [[-4., -4.], [-5., -5.], [-6., -6.]]

    results = _share_results(comm, fits_pos, fits_neg, inds)

    expected = np.array([[1., 1., -1., -1., 10.],
                         [2., 2., -2., -2., 20.],
                         [3., 3., -3., -3., 30.],
                         [4., 4., -4., -4., 40.],
                         [5., 5., -5., -5., 50.],
                         [6., 6., -6., -6., 60.]])

    assert (results == expected).all()
    # objectives = 2
    # pos = results[:, 0:objectives]
    # neg = results[:, objectives:2 * objectives]
    # inds = results[:, -1]


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    test__share_results(comm)

    print('all tests passed')
