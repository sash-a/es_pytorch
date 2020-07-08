from __future__ import annotations

import pickle
import json
from collections import namedtuple

import wandb
import numpy as np
from mpi4py import MPI

from genome import Genome
from noisetable import NoiseTable
from optimizers import Adam
import gym_runner


def percent_rank(fits: np.ndarray):
    """Transforms fitnesses into a percent of their rankings: rank/sum(ranks)"""
    assert fits.ndim == 1
    return (np.argsort(fits) + 1) / sum(range(len(fits) + 1))


def percent_fitness(fits: np.ndarray):
    """Transforms fitnesses into: fitness/total_fitness"""
    assert fits.ndim == 1
    return fits / sum(fits)


if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()

    # noinspection PyArgumentList
    cfg = json.load(open('configs/testing.json'), object_hook=lambda d: namedtuple('Cfg', d.keys())(*d.values()))

    if rank == 0:
        # noinspection PyProtectedMember
        wandb.init(project='es', entity='sash-a', name=cfg.env_name, config=dict(cfg._asdict()))

    layer_sizes = [(15, 256), (256, 256), (256, 256), (256, 3)]

    noise = NoiseTable(comm, seed=cfg.seed, table_size=cfg.table_size)
    geno = Genome(comm, layer_sizes, cfg.noise_stdev)
    n_params = len(geno.params)
    optim = Adam(geno.params, 0.01)

    for i in range(cfg.gens):
        results = []
        for _ in range(cfg.eps_per_gen):
            results.append(gym_runner.run_genome(geno, noise, cfg.env_name, cfg.max_env_steps, cfg.eps_per_policy))

        results = np.array(results * nproc)
        comm.Alltoall(results, results)

        if rank == 0:
            fits = results[:, 0]
            wandb.log({'average': np.mean(fits), 'max': np.max(fits)})
            print(f'\n\ngen:{i}\navg:{np.mean(fits)}\nmax:{np.max(fits)}\nfits:{fits}')

        fits = percent_rank(results[:, 0])
        noise_inds = results[:, 1]

        approx_grad = np.dot(fits, np.array([noise.get(int(idx), n_params) for idx in noise_inds])) / n_params
        _, geno.params = optim.update(geno.params * cfg.l2coeff - approx_grad)

        if rank == 0 and i % 1000 == 0:
            pickle.dump(geno, open(f'saved/genome-{i}', 'wb'))
