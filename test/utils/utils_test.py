import numpy as np

from es.evo.noisetable import NoiseTable
from es.utils.utils import batch_noise, scale_noise, moo_mean_rank, percent_rank


def test_batch_noise():
    table_size = 100
    params = 50
    nt = NoiseTable(params, np.arange(table_size))
    inds = np.arange(40)

    expected = [[i + j for j in range(params)] for i in range(len(inds))]

    full_batch = next(batch_noise(inds, nt, len(inds)))
    assert (full_batch == expected).all()

    batched = batch_noise(inds, nt, 19)
    assert (next(batched) == expected[:19]).all()
    assert (next(batched) == expected[19:38]).all()
    assert (next(batched) == expected[38:]).all()


def test_scale_noise():
    evals = 100
    params = 500
    table_size = 2000

    fits = np.arange(evals)
    inds = np.arange(evals)
    nt = NoiseTable(params, np.arange(table_size))

    scaled_batched = scale_noise(fits, inds, nt, 3)
    scaled_full = scale_noise(fits, inds, nt, evals)

    expected_noise = [[i + j for j in range(params)] for i in range(len(inds))]
    expected = np.dot(fits, expected_noise)

    assert (scaled_batched == expected).all()
    assert (scaled_full == expected).all()


def test_moo_mean_rank():
    evals = 10
    objectives = 2

    x = np.reshape(np.arange(evals * objectives), (-1, 2))

    mean_ranked = moo_mean_rank(x, lambda l: 2 * l)  # function that doubles the fits

    expected = []
    for fits in x:
        expected += [sum(fits)]
    assert (expected == mean_ranked).all()

    mean_ranked = moo_mean_rank(x, percent_rank)  # function that takes a percent of the fits

    res = []
    for fits in x.T:
        res.append(percent_rank(fits))

    expected = np.sum(res, axis=0) / 2

    assert (expected == mean_ranked).all()
