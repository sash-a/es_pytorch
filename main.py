from __future__ import annotations

import pickle
import numpy as np
from mpi4py import MPI

from genome import Genome
from noisetable import NoiseTable
from runner import run_genome

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()

    # TODO config
    seed = 10
    table_size = 1000000
    layer_sizes = [(4, 64), (64, 64), (64, 64), (64, 2)]
    std = 2
    lr = 0.1
    gens = 1000000

    noise = NoiseTable(comm, seed=seed, table_size=table_size)
    geno = Genome(comm, layer_sizes, std)

    for i in range(gens):
        fit, noise_idx = run_genome(geno, noise)
        results = np.array([[fit, noise_idx]] * nproc)
        comm.Alltoall(results, results)

        if rank == 0:
            fits = results[:, 0]
            print(f'\n\ngen:{i}\navg: {np.mean(fits)}\nmax:{np.max(fits)}\nfits:\n{fits}')

        # results = percentage of rank out of total rank
        # todo: should try percentage fitness of total fitness
        #  make a few methods for this
        results[:, 0] = (np.argsort(results[:, 0]) + 1) / sum(range(len(results) + 1))

        geno.approx_grad(lr, nproc, results, noise)

        if rank == 0 and i % 1000 == 0:
            pickle.dump(geno, open(f'saved/genome-{i}', 'wb'))
