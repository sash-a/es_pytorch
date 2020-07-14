import numpy as np

from noisetable import NoiseTable


def batch_noise(inds: np.ndarray, nt: NoiseTable, batch_size: int) -> np.ndarray:
    """
    Need to batch noise otherwise will have to `dot` array with shape (cfg.eps_per_gen, len(params)) or +-(5000, 136451)
    """
    assert inds.ndim == 1

    batch = []
    for idx in inds:
        batch.append(nt[int(idx)])
        if len(batch) == batch_size:
            yield np.array(batch)
            batch = []

    yield np.array(batch)


def approx_grad(fits: np.ndarray, noise_inds: np.ndarray, nt: NoiseTable, batch_size: int):
    total = 0
    batched_fits = [[fits[i + j] for i in range(batch_size)] for j in range(0, len(fits) - batch_size, batch_size)]
    batched_fits.append(fits[-(len(fits) % batch_size):])  # appending the rest that didn't make a full batch

    for fit_batch, noise_batch in zip(batched_fits, batch_noise(noise_inds, nt, batch_size)):
        total += np.dot(fit_batch, noise_batch)

    return total / nt.n_params


def percent_rank(fits: np.ndarray):
    """Transforms fitnesses into a percent of their rankings: rank/sum(ranks)"""
    assert fits.ndim == 1
    return (np.argsort(fits) + 1) / sum(range(len(fits) + 1))


def percent_fitness(fits: np.ndarray):
    """Transforms fitnesses into: fitness/total_fitness"""
    assert fits.ndim == 1
    return fits / sum(fits)
