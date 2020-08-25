from __future__ import annotations

import time
from collections.abc import Callable
from typing import List, Sequence

import gym
import numpy as np
# noinspection PyUnresolvedReferences
import pybullet_envs
from mpi4py import MPI
from numpy import ndarray
from numpy.random import RandomState
from torch.nn import Module

from es.evo.noisetable import NoiseTable
from es.evo.policy import Policy
from es.nn.optimizers import Optimizer
from es.utils.TrainingResult import TrainingResult
from es.utils.reporters import StdoutReporter, Reporter
from es.utils.utils import scale_noise, compute_centered_ranks


# noinspection PyIncorrectDocstring
def step(cfg,
         comm: MPI.Comm,
         policy: Policy,
         optim: Optimizer,
         nt: NoiseTable,
         env: gym.Env,
         fit_fn: Callable[[Module, gym.Env, int, RandomState], TrainingResult],
         rs: RandomState = np.random.RandomState(),
         rank_fn: Callable[[Sequence[ndarray]], ndarray] = compute_centered_ranks,
         reporter: Reporter = StdoutReporter(MPI.COMM_WORLD)) -> TrainingResult:
    """
    Runs a single generation of ES
    :param fit_fn: Evaluates the policy returns a TrainingResult
    :param rank_fn: Takes in fitnesses from all agents and returns those fitnesses ranked. Must be multi-objective.
    :returns: TrainingResult of the noiseless policy at that generation
    """
    assert cfg.general.eps_per_gen % comm.size == 0 and (cfg.general.eps_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.eps_per_gen / comm.size) / 2)

    gen_start = time.time()
    reporter.start_gen()
    fits_pos, fits_neg, inds = [], [], []

    for _ in range(eps_per_proc):
        idx, noise = nt.sample(rs)
        inds.append(idx)
        # for each noise ind sampled, both add and subtract the noise
        fits_pos.append(eval_one(policy, noise, fit_fn, env, cfg.env.max_steps, rs).result)
        fits_neg.append(eval_one(policy, -noise, fit_fn, env, cfg.env.max_steps, rs).result)

    objectives = len(fits_pos[0])

    results = _share_results(comm, fits_pos, fits_neg, inds)
    # subtracting rewards that used negative noise
    fits = results[:, 0:objectives] - results[:, objectives:2 * objectives]
    noise_inds = results[:, -1]

    _approx_grad(fits, noise_inds, nt, policy.flat_params, optim, rank_fn, cfg)
    noiseless_result = eval_one(policy, np.zeros(len(policy)), fit_fn, env, cfg.env.max_steps, None)
    _report(reporter, results, policy, noiseless_result, gen_start)

    return noiseless_result


def eval_one(policy: Policy, noise: ndarray, fit_fn, env: gym.Env, steps: int, rs) -> TrainingResult:
    net = policy.pheno(noise)
    return fit_fn(net, env, steps, rs)


def _share_results(comm: MPI.Comm,
                   fits_pos: List[List[float]],
                   fits_neg: List[List[float]],
                   inds: List[int]) -> ndarray:
    """share results and noise inds to all processes"""
    send_results = np.array([fp + fn + [i] for fp, fn, i in zip(fits_pos, fits_neg, inds)] * comm.size, dtype=np.float)
    results = np.empty(send_results.shape)
    comm.Alltoall(send_results, results)

    objectives = len(fits_pos[0])
    return results.reshape((-1, 1 + 2 * objectives))  # flattening the process dim


def _approx_grad(fits: ndarray, inds: ndarray, nt: NoiseTable, flat_params: ndarray, optim: Optimizer, rank_fn, cfg):
    """approximating gradient and update policy params"""
    ranked_fits = rank_fn(fits)
    grad = scale_noise(ranked_fits, inds, nt, cfg.general.batch_size) / cfg.general.eps_per_gen
    optim.step(cfg.general.l2coeff * flat_params - grad)


def _report(rep: Reporter, res: ndarray, policy: Policy, noiseless_result: TrainingResult, start: float):
    rep.report_noiseless(noiseless_result, policy)
    rep.report_fits(res[:, :-1])
    rep.end_gen(time.time() - start)
