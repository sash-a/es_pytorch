import argparse
import json
from types import SimpleNamespace

import numpy as np

from es.evo.noisetable import NoiseTable


def batch_noise(inds: np.ndarray, nt: NoiseTable, batch_size: int):
    """
    Need to batch noise otherwise will have to `dot` a large array
    """
    assert inds.ndim == 1

    batch = []
    for idx in inds:
        batch.append(nt[int(idx)])
        if len(batch) == batch_size:
            yield np.array(batch)
            del batch[:]

    if batch:
        yield np.array(batch)


def scale_noise(fits: np.ndarray, noise_inds: np.ndarray, nt: NoiseTable, batch_size: int):
    """Scales the noise according to the fitness each noise ind achieved"""
    assert len(fits) == len(noise_inds)
    total = 0
    batched_fits = [fits[i:min(i + batch_size, len(fits))] for i in range(0, len(fits), batch_size)]

    for fit_batch, noise_batch in zip(batched_fits, batch_noise(noise_inds, nt, batch_size)):
        total += np.dot(fit_batch, noise_batch)

    return total


def compute_ranks(x: np.ndarray):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x: np.ndarray):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def semi_centered_ranks(x: np.ndarray):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    s = x.size
    y = (((1 / s) * np.square(y + 0.29 * s)) / s) - 0.5
    return y


def max_normalized_ranks(x: np.ndarray):  # TODO possibly clamp the min to around -0.5
    return 2 * x / np.max(x) - 1


def signed_centered_rank(x: np.ndarray):
    return compute_ranks(np.sign(x))


def moo_mean_rank(x: np.ndarray, rank_fn):
    """
    Wrapper for rank functions to work on multi-objective fitness. Returns the mean of the ranked objectives for each
     individual.

    x: [[obj1, obj2,...]  - individual 1
        [obj1, obj2,...], - individual 2
        ... ]             - individual n
    """
    ranked = []
    for col in x.T:
        ranked.append(rank_fn(col))

    return np.mean(ranked, axis=0)


def moo_weighted_rank(x: np.ndarray, w: float, rank_fn):
    assert 0. <= w <= 1.
    assert x.shape[1] == 2  # this only works for 2 objectives

    ranked = []
    for col in x.T:
        ranked.append(rank_fn(col))

    return ranked[0] * w + ranked[1] * (1 - w)


def parse_args():
    parser = argparse.ArgumentParser(description='es-pytorch')
    parser.add_argument('config', type=str, help='Config file that will be used')
    return parser.parse_args().config


def load_config(cfg_file: str):
    """:returns: a SimpleNamespace from a json file"""
    return json.load(open(cfg_file), object_hook=lambda d: SimpleNamespace(**d))


def generate_seed(comm) -> int:
    return comm.scatter([np.random.randint(0, 1000000)] * comm.size)
