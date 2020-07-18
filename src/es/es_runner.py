from __future__ import annotations

import os
import pickle
from collections import Callable
from typing import Optional

import gym
import numpy as np
# noinspection PyUnresolvedReferences
import pybullet_envs
import torch
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
        rs: np.random.RandomState,
        rank_fn: Callable[[np.ndarray], np.ndarray],
        eval_fn: Callable[[torch.nn.Module, gym.Env, int, np.random.RandomState], float],
        reporter: Reporter = Reporter()):
    """Runs the evolutionary strategy"""

    # number of episodes per process. Total = cfg.eps_per_gen
    assert cfg.general.eps_per_gen % comm.size == 0 and (cfg.general.eps_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.eps_per_gen / comm.size) / 2)

    for gen in range(cfg.general.gens):
        fits_pos, fits_neg, inds = [], [], []
        for _ in range(eps_per_proc):
            idx, noise = nt.sample(rs)
            inds.append(idx)
            fits_pos.append(eval_single(policy, noise, eval_fn, env, cfg.env.max_steps, rs))
            fits_neg.append(eval_single(policy, -noise, eval_fn, env, cfg.env.max_steps, rs))

        # share results and noise inds to all processes
        results = np.array([fits_pos * comm.size, fits_neg * comm.size, inds * comm.size])
        comm.Alltoall(results, results)

        if comm.rank == 0:  # print results on the main process
            reporter.report_fits(gen, np.concatenate((results[:, 0], results[:, 1])))

        # approximating gradient and update policy params
        fits = rank_fn(results[:, 0] - results[:, 1])  # subtracting rewards that used negative noise
        noise_inds = results[:, 2]
        weighted_noise = scale_noise(fits, noise_inds, nt, cfg.general.batch_size)
        optim.step(weighted_noise)

        if comm.rank == 0 and gen % cfg.general.save_interval == 0 and cfg.general.save_interval > 0:  # checkpoints
            if not os.path.exists('saved'):
                os.makedirs('saved')
            pickle.dump(policy, open(f'saved/policy-{gen}', 'wb'))


def eval_single(policy: Policy, noise, eval_fn, env: gym.Env, max_steps: int, rs: Optional[np.random.RandomState]):
    net = policy.pheno(noise)
    fitness = eval_fn(net, env, max_steps, rs)
    return fitness
