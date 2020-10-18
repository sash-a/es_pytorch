import argparse
import json
from types import SimpleNamespace
from typing import Optional

import gym
import numpy as np
import torch
from mpi4py import MPI

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


def parse_args():
    parser = argparse.ArgumentParser(description='es-pytorch')
    parser.add_argument('config', type=str, help='Config file that will be used')
    return parser.parse_args().config


def load_config(cfg_file: str):
    """:returns: a SimpleNamespace from a json file"""
    return json.load(open(cfg_file), object_hook=lambda d: SimpleNamespace(**d))


def generate_seed(comm) -> int:
    return comm.scatter([np.random.randint(0, 1000000)] * comm.size)


def seed(comm: MPI.Comm, seed: int, env: Optional[gym.Env]) -> np.random.RandomState:
    """Seeds torch, the env and returns a random state that is different for each MPI proc"""
    rs = np.random.RandomState(seed + 10000 * comm.rank)  # This seed must be different on each proc
    torch.random.manual_seed(seed)  # This seed must be the same on each proc for generating initial params
    if env is not None:
        env.seed(seed)

    return rs
