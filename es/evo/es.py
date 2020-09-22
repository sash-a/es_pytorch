from __future__ import annotations

import time
from collections.abc import Callable
from typing import List, Sequence, Optional

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
from es.utils.ObStat import ObStat
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
         fit_fn: Callable[[Module, gym.Env, int, Optional[RandomState]], TrainingResult],
         rs: RandomState = np.random.RandomState(),
         rank_fn: Callable[[Sequence[ndarray]], ndarray] = compute_centered_ranks,
         reporter: Reporter = StdoutReporter(MPI.COMM_WORLD)) -> [TrainingResult, ObStat]:
    """
    Runs a single generation of ES
    :param fit_fn: Evaluates the policy returns a TrainingResult
    :param rank_fn: Takes in fitnesses from all agents and returns those fitnesses ranked. Must be multi-objective.
    :returns: TrainingResult of the noiseless policy at that generation
    """
    assert cfg.general.policies_per_gen % comm.size == 0 and (cfg.general.policies_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.policies_per_gen / comm.size) / 2)

    gen_start = time.time()
    gen_obstat = ObStat(env.observation_space.shape, 0)
    reporter.start_gen()
    results_pos, results_neg, inds = [], [], []

    for _ in range(eps_per_proc):
        idx, noise = nt.sample(rs)
        inds.append(idx)
        # for each noise ind sampled, both add and subtract the noise
        results_pos.append(fit_fn(policy.pheno(noise), env, cfg.env.max_steps, rs))
        results_neg.append(fit_fn(policy.pheno(-noise), env, cfg.env.max_steps, rs))
        gen_obstat.inc(*results_pos[-1].ob_sum_sq_cnt)
        gen_obstat.inc(*results_neg[-1].ob_sum_sq_cnt)

    n_objectives = len(results_pos[0].result)

    results = _share_results(comm, [tr.result for tr in results_pos], [tr.result for tr in results_neg], inds)
    fits_pos, fits_neg = results[:, 0:n_objectives], results[:, n_objectives:2 * n_objectives]
    ranked_fits = rank_fn(np.concatenate((fits_pos, fits_neg)))
    ranked_fits = ranked_fits[:len(fits_pos)] - ranked_fits[len(fits_pos):]
    noise_inds = results[:, -1]
    steps = comm.allreduce(sum([tr.steps for tr in results_pos + results_neg]), op=MPI.SUM)
    gen_obstat.mpi_inc(comm)

    _approx_grad(ranked_fits, noise_inds, nt, policy.flat_params, optim, cfg)
    noiseless_result = fit_fn(policy.pheno(np.zeros(len(policy))), env, cfg.env.max_steps, rs)
    reporter.end_gen(np.concatenate((fits_pos, fits_neg), 0), noiseless_result, policy, steps, time.time() - gen_start)

    return noiseless_result, gen_obstat


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


def _approx_grad(ranked_fits: ndarray, inds: ndarray, nt: NoiseTable, flat_params: ndarray, optim: Optimizer, cfg):
    """approximating gradient and update policy params"""
    grad = scale_noise(ranked_fits, inds, nt, cfg.general.batch_size) / cfg.general.policies_per_gen
    optim.step(cfg.policy.l2coeff * flat_params - grad)
