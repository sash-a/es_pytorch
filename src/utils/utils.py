import argparse
import json
from collections import namedtuple
from typing import List

import numpy as np

from es.noisetable import NoiseTable


def batch_noise(inds: np.ndarray, signs: List[int], nt: NoiseTable, batch_size: int):
    """
    Need to batch noise otherwise will have to `dot` array with shape (cfg.eps_per_gen, len(params)) or +-(5000, 136451)
    """
    assert inds.ndim == 1

    batch = []
    for idx, sign in zip(inds, signs):
        batch.append(sign * nt[int(idx)])
        if len(batch) == batch_size:
            yield np.array(batch)
            batch = []

    yield np.array(batch)


def scale_noise(fits: np.ndarray, noise_inds: np.ndarray, signs: List[int], nt: NoiseTable, batch_size: int):
    """Scales the noise according to the fitness each noise ind achieved"""
    batched_fits = [[fits[i + j] for i in range(batch_size)] for j in range(0, len(fits) - batch_size, batch_size)]
    leftover = len(fits) % batch_size  # appending the rest that didn't make a full batch
    if leftover != 0:
        batched_fits.append(fits[-leftover:])

    total = 0

    for fit_batch, noise_batch in zip(batched_fits, batch_noise(noise_inds, signs, nt, batch_size)):
        total += np.dot(fit_batch, noise_batch)

    return total


def percent_rank(fits: np.ndarray):
    """Transforms fitnesses into a percent of their rankings: rank/sum(ranks)"""
    assert fits.ndim == 1
    return (np.argsort(fits) + 1) / sum(range(len(fits) + 1))


def percent_fitness(fits: np.ndarray):
    """Transforms fitnesses into: fitness/total_fitness"""
    assert fits.ndim == 1
    return fits / sum(fits)


def parse_args():
    parser = argparse.ArgumentParser(description='es-pytorch')
    parser.add_argument('config', type=str, help='Config file that will be used')
    return parser.parse_args().config


def load_config(cfg_file: str):
    """:returns: named tuple"""
    # noinspection PyArgumentList
    return json.load(open(cfg_file), object_hook=lambda d: namedtuple('Cfg', d.keys())(*d.values()))