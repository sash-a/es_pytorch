import pytest


@pytest.fixture
def comm():
    from mpi4py import MPI

    return MPI.COMM_WORLD