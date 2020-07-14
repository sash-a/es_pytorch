from __future__ import annotations

import pickle
import json
from collections import namedtuple
from typing import Callable

import gym
# noinspection PyUnresolvedReferences
import pybullet_envs

import torch
# import wandb
import numpy as np
from mpi4py import MPI

from nn_structures import FullyConnected
from noisetable import NoiseTable
from utils import approx_grad, percent_rank
from optimizers import Adam
import gym_runner
from policy import Policy


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
    nt: NoiseTable = NoiseTable.create_shared_noisetable(comm, cfg.table_size, len(policy), cfg.seed)
    env = gym.make(cfg.env_name)

    for gen in range(cfg.gens):
        results = []
        for _ in range(int(cfg.eps_per_gen / size)):  # evaluate policies
            net, idx = policy.pheno(nt)
            fitness = gym_runner.run_model(net, env, cfg.max_env_steps)
            results += [(fitness, idx)]

        # share results and noise inds to all processes
        results = np.array(results * size)
        comm.Alltoall(results, results)

        if rank == 0:
            # print results
            fits = results[:, 0]
            avg = np.mean(fits)
            mx = np.max(fits)

            # wandb.log({'average': avg, 'max': mx})

        # approximating gradient and updating policy params
        fits = percent_rank(results[:, 0])
        noise_inds = results[:, 1]
        grad = approx_grad(fits, noise_inds, nt, cfg.batch_size)
        _, policy.flat_params = optim.update(policy.flat_params * cfg.l2coeff - grad)

        if rank == 0 and gen % cfg.save_interval == 0:  # saving policy
            pickle.dump(policy, open(f'saved/policy-{gen}', 'wb'))

    env.close()
