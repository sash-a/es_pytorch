from __future__ import annotations

import gc
import os
import pickle
from collections import Callable

# noinspection PyUnresolvedReferences
import pybullet_envs
import gym
import torch
import numpy as np

from mpi4py import MPI

from es.noisetable import NoiseTable
from es.optimizers import ES
from es.policy import Policy
from utils.reporter import Reporter
from utils.utils import scale_noise


def run(cfg,
        comm: MPI.Comm,
        policy: Policy,
        optim: ES,  # TODO make generic optimizer
        nt: NoiseTable,
        env: gym.Env,
        rank_fn: Callable[[np.ndarray], np.ndarray],
        eval_fn: Callable[[torch.nn.Module, gym.Env, int], float],
        reporter: Reporter = Reporter()):
    """Runs the evolutionary strategy"""

    assert cfg.eps_per_gen % comm.size == 0 and (cfg.eps_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.eps_per_gen / comm.size) / 2)  # number of episodes per process. Total = cfg.eps_per_gen

    for gen in range(cfg.gens):
        results = []
        for _ in range(eps_per_proc):
            # Both add and subtract the same noise
            idx, noise = nt.sample()
            for i in [-1, 1]:
                net = policy.pheno(noise, i)
                fitness = eval_fn(net, env, cfg.max_env_steps)
                results += [(fitness, idx)]

        # Hopefully freeing up this memory
        del noise
        gc.collect()

        # share results and noise inds to all processes
        results = np.array(results * comm.size)
        comm.Alltoall(results, results)

        if comm.rank == 0:  # print results on the main process
            reporter.report_fits(gen, results[:, 0])

        # approximating gradient and updating policy params
        fits = rank_fn(results[:, 0])
        noise_inds = results[:, 1]
        signs = [-1, 1] * len(noise_inds)  # denotes if each noise ind was added or subtracted
        weighted_noise = scale_noise(fits, noise_inds, signs, nt, cfg.batch_size)
        optim.step(weighted_noise)

        if comm.rank == 0 and gen % cfg.save_interval == 0 and cfg.save_interval > 0:  # saving policy
            if not os.path.exists('saved'):
                os.makedirs('saved')
            pickle.dump(policy, open(f'saved/policy-{gen}', 'wb'))
