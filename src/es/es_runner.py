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
from numpy.random import RandomState

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
        rs: RandomState,
        rank_fn: Callable[[np.ndarray], np.ndarray],
        fit_fn: Callable[[torch.nn.Module, gym.Env, int, RandomState], float],
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
            fits_pos.append(eval_one(policy, noise, fit_fn, env, cfg.env.max_steps, rs))
            fits_neg.append(eval_one(policy, -noise, fit_fn, env, cfg.env.max_steps, rs))

        # share results and noise inds to all processes
        send_results = np.array([[fp, fn, i] for fp, fn, i in zip(fits_pos, fits_neg, inds)] * comm.size)
        results = np.empty(send_results.shape)
        comm.Alltoall(send_results, results)

        # approximating gradient and update policy params
        fits = rank_fn(results[:, 0] - results[:, 1])  # subtracting rewards that used negative noise
        noise_inds = results[:, 2]
        weighted_noise = scale_noise(fits, noise_inds, nt, cfg.general.batch_size)
        optim.step(weighted_noise)

        if comm.rank == 0:
            noiseless_fit = eval_one(policy, np.zeros(len(policy)), fit_fn, env, cfg.env.max_steps, None)
            reporter.report_fits(gen, np.concatenate((results[:, 0], results[:, 1])))
            reporter.report_noiseless(gen, noiseless_fit)
            if gen % cfg.general.save_interval == 0 and cfg.general.save_interval > 0:  # checkpoints
                if not os.path.exists('saved'):
                    os.makedirs('saved')
                pickle.dump(policy, open(f'saved/policy-{gen}', 'wb'))


def eval_one(policy: Policy, noise: np.ndarray, fit_fn, env: gym.Env, steps: int, rs: Optional[RandomState]) -> float:
    net = policy.pheno(noise)
    fitness = fit_fn(net, env, steps, rs)
    return fitness
