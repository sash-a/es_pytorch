# Very informal test to make sure that parallelism is working

import numpy as np

from src.core.es import _share_results
# noinspection PyUnresolvedReferences
from test import comm


def test__share_results(comm):
    evals = 5
    objectives = 4
    proc_factor = evals * comm.rank + 1

    inds = (np.arange(evals) + proc_factor) * 10

    fits_pos = [
        [i + i * 10 ** j if j != 0 else i for j in range(objectives)] for i in range(proc_factor, proc_factor + evals)
    ]
    fits_neg = (-np.array(fits_pos)).tolist()

    results = _share_results(comm, fits_pos, fits_neg, inds)

    expected = []
    for i in range(1, evals * comm.size + 1):
        pos = [i + i * 10 ** j if j != 0 else i for j in range(objectives)]
        neg = (-np.array(pos)).tolist()
        res = pos + neg + [i * 10]
        expected.append(res)

    assert (results == expected).all()
