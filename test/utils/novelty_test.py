# Very informal test to make sure that parallelism is working
import numpy as np

from es.utils.novelty import update_archive, novelty
# noinspection PyUnresolvedReferences
from test import comm


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


def test_novelty():
    beh = np.array([0, 0])
    archive = np.array([[2, 2], [1, 1], [3, 3]])
    assert novelty(beh, archive, 1) == 2
    assert novelty(beh, archive, 2) == 10 / 2
    assert novelty(beh, archive, 3) == 28 / 3
    assert novelty(beh, archive, 50) == 28 / 3
