from __future__ import annotations

import argparse
import json
import pickle
from collections import namedtuple

import gym
# import wandb
import numpy as np
# noinspection PyUnresolvedReferences
import pybullet_envs
import torch
from mpi4py import MPI

import gym_runner
from nn_structures import FullyConnected
from noisetable import NoiseTable
from optimizers import Adam
from policy import Policy
from utils import approx_grad, percent_rank

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='es-pytorch')
    parser.add_argument('config', type=str, help='Config file that will be used')
    cfg_file = parser.parse_args().config

    comm: MPI.Comm = MPI.COMM_WORLD
    # noinspection PyArgumentList
    cfg = json.load(open(cfg_file), object_hook=lambda d: namedtuple('Cfg', d.keys())(*d.values()))
    # if rank == 0:
    #     # noinspection PyProtectedMember
    #     wandb.init(project='es', entity='sash-a', name=cfg.env_name, config=dict(cfg._asdict()))

    policy: Policy = Policy(FullyConnected(15, 3, 256, 2, torch.nn.Tanh), cfg.noise_stdev)
    optim: Adam = Adam(policy.flat_params, cfg.lr)
    nt: NoiseTable = NoiseTable.create_shared_noisetable(comm, cfg.table_size, len(policy), cfg.seed)
    env: gym.Env = gym.make(cfg.env_name)

    for gen in range(cfg.gens):
        results = []
        for _ in range(int(cfg.eps_per_gen / comm.size)):  # evaluate policies
            net, idx = policy.pheno(nt)
            fitness = gym_runner.run_model(net, env, cfg.max_env_steps, cfg.eps_per_policy)
            results += [(fitness, idx)]

        # share results and noise inds to all processes
        results = np.array(results * comm.size)
        comm.Alltoall(results, results)

        if comm.rank == 0:  # print results on the main process
            fits = results[:, 0]
            avg = np.mean(fits)
            mx = np.max(fits)
            print(f'Gen:{gen}-avg:{avg:0.2f}-max:{mx:0.2f}')

            # wandb.log({'average': avg, 'max': mx})

        # approximating gradient and updating policy params
        fits = percent_rank(results[:, 0])
        noise_inds = results[:, 1]
        grad = approx_grad(fits, noise_inds, nt, cfg.batch_size)
        _, policy.flat_params = optim.update(policy.flat_params * cfg.l2coeff - grad)

        if comm.rank == 0 and gen % cfg.save_interval == 0:  # saving policy
            pickle.dump(policy, open(f'saved/policy-{gen}', 'wb'))

    env.close()
