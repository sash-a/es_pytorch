from __future__ import annotations

import argparse
import gc
import json
import pickle
from collections import namedtuple

# import wandb
import numpy as np
import torch
from mpi4py import MPI

# noinspection PyUnresolvedReferences
import pybullet_envs
import gym
import gym_runner

from nn_structures import FullyConnected
from noisetable import NoiseTable
from utils import scale_noise, percent_rank
from optimizers import ES
from policy import Policy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='es-pytorch')
    parser.add_argument('config', type=str, help='Config file that will be used')
    cfg_file = parser.parse_args().config

    comm: MPI.Comm = MPI.COMM_WORLD
    # noinspection PyArgumentList
    cfg = json.load(open(cfg_file), object_hook=lambda d: namedtuple('Cfg', d.keys())(*d.values()))

    assert cfg.eps_per_gen % comm.size == 0 and (cfg.eps_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.eps_per_gen / comm.size) / 2)  # number of episodes per process. Total = eps_per_gen

    # if rank == 0:
    #     # noinspection PyProtectedMember
    #     wandb.init(project='es', entity='sash-a', name=cfg.env_name, config=dict(cfg._asdict()))

    torch.random.manual_seed(cfg.seed)
    policy: Policy = Policy(FullyConnected(15, 3, 256, 2, torch.nn.Tanh), cfg.noise_stdev)
    optim: ES = ES(policy, cfg.lr, cfg.eps_per_gen)

    nt: NoiseTable = NoiseTable.create_shared_noisetable(comm, cfg.table_size, len(policy), cfg.seed)
    env: gym.Env = gym.make(cfg.env_name)

    for gen in range(cfg.gens):
        results = []
        for _ in range(eps_per_proc):
            # Both add and subtract the same noise
            idx, noise = nt.sample()
            for i in [-1, 1]:
                net = policy.pheno(noise, i)
                fitness = gym_runner.run_model(net, env, cfg.max_env_steps)
                results += [(fitness, idx)]

        # Hopefully freeing up this memory
        del noise
        gc.collect()

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
        signs = [-1, 1] * len(noise_inds)  # denotes if each noise ind was added or subtracted
        weighted_noise = scale_noise(fits, noise_inds, signs, nt, cfg.batch_size)
        optim.step(weighted_noise)

        if comm.rank == 0 and gen % cfg.save_interval == 0:  # saving policy
            pickle.dump(policy, open(f'saved/policy-{gen}', 'wb'))

    env.close()
