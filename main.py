from __future__ import annotations

import pickle
import json
from collections import namedtuple

import gym
# noinspection PyUnresolvedReferences
import pybullet_envs

import torch
# import wandb
import numpy as np
from mpi4py import MPI

from nn_structures import FullyConnected
from noisetable import NoiseTable
from optimizers import Adam
import gym_runner
from policy import Policy


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
    size = comm.size
    rank = comm.rank
    # noinspection PyArgumentList
    cfg = json.load(open('configs/testing.json'), object_hook=lambda d: namedtuple('Cfg', d.keys())(*d.values()))
    # if rank == 0:
    #     # noinspection PyProtectedMember
    #     wandb.init(project='es', entity='sash-a', name=cfg.env_name, config=dict(cfg._asdict()))

    policy: Policy = Policy(FullyConnected(15, 3, 256, 2, torch.nn.Tanh), cfg.noise_stdev)
    optim: Adam = Adam(policy.flat_params, 0.01)
    nt: NoiseTable = NoiseTable(comm, cfg.table_size, len(policy), cfg.seed)
    env = gym.make(cfg.env_name)

    for gen in range(cfg.gens):
        results = []
        for _ in range(int(cfg.eps_per_gen / size)):
            net, idx = policy.pheno(nt)
            fitness = gym_runner.run_model(net, env, cfg.max_env_steps)
            results += [(fitness, idx)]

        results = np.array(results * size)
        comm.Alltoall(results, results)

        if rank == 0:
            fits = results[:, 0]
            avg = np.mean(fits)
            mx = np.max(fits)

            # wandb.log({'average': avg, 'max': mx})
            print(f'gen:{gen}\navg:{avg}\nmax:{mx}')

        fits = percent_rank(results[:, 0])
        noise_inds = results[:, 1]

        approx_grad = np.dot(fits, np.array([nt[int(i)] for i in noise_inds])) / policy.n_params
        _, policy.flat_params = optim.update(policy.flat_params * cfg.l2coeff - approx_grad)

        if rank == 0 and gen % cfg.save_interval == 0:
            pickle.dump(policy, open(f'saved/policy-{gen}', 'wb'))

    env.close()
