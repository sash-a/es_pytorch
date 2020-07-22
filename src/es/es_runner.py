from __future__ import annotations

import time
from collections.abc import Callable
from typing import Optional

import gym
import numpy as np
# noinspection PyUnresolvedReferences
import pybullet_envs
import torch
from mpi4py import MPI
from numpy.random import RandomState

from es.noisetable import NoiseTable
from es.optimizers import Optimizer
from es.policy import Policy
from utils.gym_runner import run_model
from utils.reporters import StdoutReporter, Reporter
from utils.utils import scale_noise, compute_ranks


def run(cfg,
        comm: MPI.Comm,
        policy: Policy,
        optim: Optimizer,
        nt: NoiseTable,
        env: gym.Env,
        rs: RandomState = np.random.RandomState(),
        rank_fn: Callable[[np.ndarray], np.ndarray] = compute_ranks,
        fit_fn: Callable[[torch.nn.Module, gym.Env, int, RandomState], float] = run_model,
        reporter: Reporter = StdoutReporter(MPI.COMM_WORLD)):
    """Runs the evolutionary strategy"""

    # number of episodes per process. Total = cfg.eps_per_gen
    assert cfg.general.eps_per_gen % comm.size == 0 and (cfg.general.eps_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.eps_per_gen / comm.size) / 2)

    for gen in range(cfg.general.gens):
        gen_start = time.time()
        reporter.start_gen(gen)
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
        grad = scale_noise(fits, noise_inds, nt, cfg.general.batch_size) / cfg.general.eps_per_gen
        optim.step(cfg.general.l2coeff * policy.flat_params - grad)

        if comm.rank == 0:
            reporter.report_noiseless(eval_one(policy, np.zeros(len(policy)), fit_fn, env, cfg.env.max_steps, None))

        reporter.report_fits(np.concatenate((results[:, 0], results[:, 1])))
        reporter.end_gen(time.time() - gen_start, policy)


def eval_one(policy: Policy, noise: np.ndarray, fit_fn, env: gym.Env, steps: int, rs: Optional[RandomState]) -> float:
    net = policy.pheno(noise)
    fitness = fit_fn(net, env, steps, rs)
    return fitness
