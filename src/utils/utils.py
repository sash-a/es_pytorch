import argparse
import heapq
import json
from collections import namedtuple

import numpy as np
from scipy.spatial import distance

from es.noisetable import NoiseTable


def novelty(behaviour: np.ndarray, others: np.ndarray, n: int) -> float:
    dists = heapq.nsmallest(n, distance.cdist(behaviour, others, 'sqeuclidean')[0])
    return sum(dists)


def batch_noise(inds: np.ndarray, nt: NoiseTable, batch_size: int):
    """
    Need to batch noise otherwise will have to `dot` array with shape (cfg.eps_per_gen, len(params)) or +-(5000, 136451)
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


def percent_rank(fits: np.ndarray):
    """Transforms fitnesses into a percent of their rankings: rank/sum(ranks)"""
    assert fits.ndim == 1
    ranked = compute_ranks(fits) + 1
    return ranked / sum(range(len(ranked) + 1))


def percent_fitness(fits: np.ndarray):
    """Transforms fitnesses into: fitness/total_fitness"""
    assert fits.ndim == 1
    return fits / sum(fits)


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def parse_args():
    parser = argparse.ArgumentParser(description='es-pytorch')
    parser.add_argument('config', type=str, help='Config file that will be used')
    return parser.parse_args().config


def load_config(cfg_file: str):
    """:returns: named tuple"""
    # noinspection PyArgumentList
    return json.load(open(cfg_file), object_hook=lambda d: namedtuple('Cfg', d.keys())(*d.values()))
