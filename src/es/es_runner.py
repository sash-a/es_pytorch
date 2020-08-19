from __future__ import annotations

import time
from collections.abc import Callable
from typing import Optional, List, Tuple

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
from utils.reporters import StdoutReporter, Reporter
from utils.utils import scale_noise, compute_ranks


def step(cfg,
         comm: MPI.Comm,
         policy: Policy,
         optim: Optimizer,
         nt: NoiseTable,
         env: gym.Env,
         fit_fn: Callable[[torch.nn.Module, gym.Env, int, RandomState], Tuple[float, dict]],
         rs: RandomState = np.random.RandomState(),
         rank_fn: Callable[[np.ndarray], np.ndarray] = compute_ranks,
         reporter: Reporter = StdoutReporter(MPI.COMM_WORLD)):
    """Runs a single generation of ES"""
    assert cfg.general.eps_per_gen % comm.size == 0 and (cfg.general.eps_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.eps_per_gen / comm.size) / 2)

    for gen in range(cfg.general.gens):
        gen_start = time.time()
        reporter.start_gen(gen)
        fits_pos, fits_neg, inds = [], [], []

        for _ in range(eps_per_proc):
            idx, noise = nt.sample(rs)
            inds.append(idx)
            # for each noise ind sampled, both add and subtract the noise
            fits_pos.append(eval_one(policy, noise, fit_fn, env, cfg.env.max_steps, rs)[0])
            fits_neg.append(eval_one(policy, -noise, fit_fn, env, cfg.env.max_steps, rs)[0])

        results = _share_results(comm, fits_pos, fits_neg, inds)
        _approx_grad(results, nt, policy.flat_params, optim, rank_fn, cfg)
        _report(comm, results, policy, fit_fn, env, cfg, reporter, gen_start)


def eval_one(policy: Policy, noise: np.ndarray, fit_fn, env: gym.Env, steps: int, rs) -> Tuple[float, dict]:
    net = policy.pheno(noise)
    fitness, info = fit_fn(net, env, steps, rs)
    return fitness, info


def _share_results(comm: MPI.Comm, fits_pos: List[float], fits_neg: List[float], inds: List[int]) -> np.ndarray:
    # share results and noise inds to all processes
    send_results = np.array([list(zip(fits_pos, fits_neg, inds))] * comm.size)
    results = np.empty(send_results.shape)
    comm.Alltoall(send_results, results)
    return results.reshape((-1, 3))  # flattening the process dim


def _approx_grad(results: np.ndarray, nt: NoiseTable, flat_params: np.ndarray, optim: Optimizer, rank_fn, cfg):
    # approximating gradient and update policy params
    fits = rank_fn(results[:, 0] - results[:, 1])  # subtracting rewards that used negative noise
    noise_inds = results[:, 2]
    grad = scale_noise(fits, noise_inds, nt, cfg.general.batch_size) / cfg.general.eps_per_gen
    optim.step(cfg.general.l2coeff * flat_params - grad)


def _report(comm: MPI.Comm, res: np.ndarray, policy: Policy, fit_fn, env: gym.Env, cfg, rep: Reporter, start: float):
    fitness, info = 0, {}
    if comm.rank == 0:
        fitness, info = eval_one(policy, np.zeros(len(policy)), fit_fn, env, cfg.env.max_steps, None)

    rep.report_noiseless(fitness, info)
    rep.report_fits(np.concatenate((res[:, 0], res[:, 1])))
    rep.end_gen(time.time() - start, policy)
