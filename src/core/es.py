from __future__ import annotations

from collections.abc import Callable
from typing import List, Tuple

import gym
import numpy as np
# noinspection PyUnresolvedReferences
import pybullet_envs
from mpi4py import MPI
from numpy import ndarray
from numpy.random import RandomState
from torch.nn import Module

from src.core.noisetable import NoiseTable
from src.core.policy import Policy
from src.gym.training_result import TrainingResult
from src.nn.obstat import ObStat
from src.nn.optimizers import Optimizer
from src.utils.rankers import Ranker, CenteredRanker
from src.utils.reporters import StdoutReporter, Reporter
from src.utils.utils import scale_noise


# noinspection PyIncorrectDocstring
def step(cfg,
         comm: MPI.Comm,
         policy: Policy,
         optim: Optimizer,
         nt: NoiseTable,
         env: gym.Env,
         fit_fn: Callable[[Module], TrainingResult],
         rs: RandomState = np.random.RandomState(),
         ranker: Ranker = CenteredRanker,
         reporter: Reporter = StdoutReporter(MPI.COMM_WORLD)) -> [TrainingResult, ObStat]:
    """
    Runs a single generation of ES
    :param fit_fn: Evaluates the policy returns a :class:`TrainingResult`
    :param ranker: A subclass of :class:`Ranker` that is able to rank the fitnesses
    :returns: :class:`TrainingResult` of the noiseless policy at that generation
    """
    assert cfg.general.policies_per_gen % comm.size == 0 and (cfg.general.policies_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.policies_per_gen / comm.size) / 2)

    gen_obstat = ObStat(env.observation_space.shape, 0)
    pos_res, neg_res, inds, steps = test_params(comm, eps_per_proc, policy, nt, gen_obstat, fit_fn, rs)

    reporter.print(f'n dupes: {len(inds) - len(set(inds))}')

    ranker.rank(pos_res, neg_res, inds)
    approx_grad(ranker, nt, policy.flat_params, optim, cfg.general.batch_size, cfg.policy.l2coeff)
    noiseless_result = fit_fn(policy.pheno(np.zeros(len(policy))), False)
    reporter.log_gen(ranker.fits, noiseless_result, policy, steps)

    return noiseless_result, gen_obstat


def test_params(comm: MPI.Comm, n: int, policy: Policy, nt: NoiseTable, gen_obstat: ObStat,
                fit_fn: Callable[[Module], TrainingResult], rs: RandomState) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Tests `n` different perturbations of `policy`'s params and returns the positive and negative results
    (from all processes).

    Where positive_result[i] is the fitness when the noise at nt[noise_inds[i]] is added to policy.flat_params
    and negative_result[i] is when the same noise is subtracted

    :returns: tuple(positive results, negative results, noise inds, total steps)
    """
    results_pos, results_neg, inds = [], [], []
    for _ in range(n):
        idx, noise = nt.sample(rs)
        inds.append(idx)
        # for each noise ind sampled, both add and subtract the noise
        results_pos.append(fit_fn(policy.pheno(noise)))
        results_neg.append(fit_fn(policy.pheno(-noise)))
        gen_obstat.inc(*results_pos[-1].ob_sum_sq_cnt)
        gen_obstat.inc(*results_neg[-1].ob_sum_sq_cnt)

    n_objectives = len(results_pos[0].result)
    results = _share_results(comm, [tr.result for tr in results_pos], [tr.result for tr in results_neg], inds)
    gen_obstat.mpi_inc(comm)
    steps = comm.allreduce(sum([tr.steps for tr in results_pos + results_neg]), op=MPI.SUM)

    return results[:, 0:n_objectives], results[:, n_objectives:2 * n_objectives], results[:, -1], steps


def _share_results(comm: MPI.Comm,
                   fits_pos: List[List[float]],
                   fits_neg: List[List[float]],
                   inds: List[int]) -> ndarray:
    """Share results and noise inds to all processes"""
    send_results = np.array([fp + fn + [i] for fp, fn, i in zip(fits_pos, fits_neg, inds)] * comm.size, dtype=np.float)
    results = np.empty(send_results.shape)
    comm.Alltoall(send_results, results)

    objectives = len(fits_pos[0])

    return results.reshape((-1, 1 + 2 * objectives))  # flattening the process dim


def approx_grad(ranker: Ranker, nt: NoiseTable, params: ndarray, optim: Optimizer, batch_size: int, l2coeff: float):
    """Approximating gradient and update policy params"""
    grad = scale_noise(ranker.ranked_fits, ranker.noise_inds, nt, batch_size) / ranker.n_fits_ranked
    optim.step(l2coeff * params - grad)
