# Very informal test to make sure that parallelism is working
import numpy as np
import pytest

from src.utils.novelty import update_archive


@pytest.fixture
def comm():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def test_update_archive(comm):
    if comm.rank == 0:
        beh = [1., 2., 3.]
    else:
        beh = [4., 3., 50.]

    archive = update_archive(comm, beh, None)
    assert (archive == np.array([[1., 2., 3.]])).all()

    if comm.rank == 0:
        beh = [10., 20., 30.]
    else:
        beh = [111., 222., 333.]

    archive = update_archive(comm, beh, archive)
    assert (archive == np.array([[1., 2., 3.], [10., 20., 30.]])).all()
