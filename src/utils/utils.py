import argparse
import json
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from mpi4py import MPI
from munch import Munch, munchify

from src.core.noisetable import NoiseTable


def batch_noise(inds: np.ndarray, nt: NoiseTable, policy_len: int, batch_size: int):
    """Need to batch noise otherwise will have to `dot` a large array"""
    assert inds.ndim == 1

    batch = []
    for idx in inds:
        batch.append(nt.get(int(idx), policy_len))
        if len(batch) == batch_size:
            yield np.array(batch)
            del batch[:]

    if batch:
        yield np.array(batch)


def scale_noise(fits: np.ndarray, noise_inds: np.ndarray, nt: NoiseTable, policy_len: int, batch_size: int):
    """Scales the noise according to the fitness each noise ind achieved"""
    assert len(fits) == len(noise_inds)
    total = 0
    batched_fits = [fits[i:min(i + batch_size, len(fits))] for i in range(0, len(fits), batch_size)]
    comm = MPI.COMM_WORLD

    for fit_batch, noise_batch in zip(batched_fits, batch_noise(noise_inds, nt, policy_len, batch_size)):
        total += np.dot(fit_batch, noise_batch)

    return total


def parse_args():
    parser = argparse.ArgumentParser(description='es-pytorch')
    parser.add_argument('config', type=str, help='Config file that will be used')
    return parser.parse_args().config


def load_config(cfg_file: str) -> Munch:
    """:returns: a Munch from a json file"""
    with open(cfg_file) as f:
        d = json.load(f)

    return munchify(d)


def generate_seed(comm) -> int:
    """Not a good method to generate a seed, rather use: gym.utils.seeding.np_random"""
    return comm.scatter([np.random.randint(0, 1000000)] * comm.size)


def seed(comm: MPI.Comm, seed: list, env: Optional[gym.Env] = None) -> Tuple[np.random.RandomState, int, int]:
    """Seeds torch, the env and returns the seed and a random state"""
    if seed is not None and hasattr(seed, '__len__') and len(seed) == comm.size:
        my_seed = seed[comm.rank]
        rs = np.random.RandomState(my_seed)
    else:
        rs, my_seed = gym.utils.seeding.np_random(None)

    global_seed = comm.scatter([my_seed] * comm.size)  # scatter root procs `my_seed` for seeding torch
    torch.random.manual_seed(global_seed)  # This seed must be the same on each proc for generating initial params
    if env is not None:
        env.seed(my_seed)
        env.action_space.seed(my_seed)
        env.observation_space.seed(my_seed)

    return rs, my_seed, global_seed
