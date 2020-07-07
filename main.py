from __future__ import annotations

import pickle
import numpy as np
from mpi4py import MPI

from genome import Genome
from noisetable import NoiseTable
from optimizers import Adam
from runner import run_genome


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

    # TODO config
    seed = 10
    table_size = 10000000

    layer_sizes = [(15, 128), (128, 128), (128, 128), (128, 3)]

    std = 2

    lr = 0.01
    l2coeff = 0.005

    gens = 10000000
    episodes = 10

    noise = NoiseTable(comm, seed=seed, table_size=table_size)
    geno = Genome(comm, layer_sizes, std)
    n_params = len(geno.params)
    optim = Adam(geno.params, 0.01)

    for i in range(gens):
        # TODO episodes
        fit, noise_idx = run_genome(geno, noise, 'HopperBulletEnv-v0')

        results = np.array([[fit, noise_idx]] * nproc)
        comm.Alltoall(results, results)

        if rank == 0:
            fits = results[:, 0]
            print(f'\n\ngen:{i}\navg:{np.mean(fits)}\nmax:{np.max(fits)}\nfits:{fits}')

        # results = percentage of rank out of total rank
        # todo: should try percentage fitness of total fitness
        #  make a few methods for this
        fits = percent_rank(results[:, 0])
        noise_inds = results[:, 1]

        approx_grad = np.dot(fits, np.array([noise.get(idx, n_params) for idx in noise_inds])) / n_params
        _, geno.params = optim.update(geno.params * l2coeff - approx_grad)

        if rank == 0 and i % 1000 == 0:
            pickle.dump(geno, open(f'saved/genome-{i}', 'wb'))
