from __future__ import annotations

import time
from collections.abc import Callable
from typing import List, Optional, Tuple

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
from es.utils.obstat import ObStat
from es.utils.ranking_functions import Ranker, CenteredRanker
from es.utils.reporters import StdoutReporter, Reporter
from es.utils.training_result import TrainingResult, SavingResult
from es.utils.utils import scale_noise


# noinspection PyIncorrectDocstring
def step(cfg,
         comm: MPI.Comm,
         policy: Policy,
         optim: Optimizer,
         nt: NoiseTable,
         env: gym.Env,
         fit_fn: Callable[[Module, gym.Env, int, Optional[RandomState]], TrainingResult],
         global_best: SavingResult,
         local_best: SavingResult,
         rs: RandomState = np.random.RandomState(),
         ranker: Ranker = CenteredRanker,
         reporter: Reporter = StdoutReporter(MPI.COMM_WORLD)) -> [TrainingResult, ObStat]:
    """
    Runs a single generation of ES

    :param fit_fn: Evaluates the policy returns a TrainingResult
    :param ranker: A subclass of `Ranker` that is able to rank the fitnesses
    :returns: TrainingResult of the noiseless policy at that generation
    """
    assert cfg.general.policies_per_gen % comm.size == 0 and (cfg.general.policies_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.policies_per_gen / comm.size) / 2)

    gen_start = time.time()  # TODO make this internal to reporters?
    gen_obstat = ObStat(env.observation_space.shape, 0)
    results_pos, results_neg, inds, ws = [], [], [], []

    for _ in range(eps_per_proc):
        idx, noise = nt.sample(rs, nt.n_params + 1)
        ws.append(noise[-1] * cfg.experimental.w_std)
        noise = noise[:-1]
        inds.append(idx)
        # for each noise ind sampled, both add and subtract the noise
        results_pos.append(fit_fn(policy.pheno(noise), env, cfg.env.max_steps, rs))
        results_neg.append(fit_fn(policy.pheno(-noise), env, cfg.env.max_steps, rs))
        gen_obstat.inc(*results_pos[-1].ob_sum_sq_cnt)
        gen_obstat.inc(*results_neg[-1].ob_sum_sq_cnt)

    n_objectives = len(results_pos[0].result)

    results, ws = _share_results(comm, [tr.result for tr in results_pos], [tr.result for tr in results_neg], inds, ws)
    ranked, extra_ranked = ranker.rank(results[:, 0:n_objectives],
                                       results[:, n_objectives:2 * n_objectives],
                                       results[:, -1],
                                       # ws=np.repeat(np.clip(policy.w + ws, 0, 1), 2),
                                       lbest=local_best,
                                       gbest=global_best)

    noise_mags = []
    for i in ranker.noise_inds:
        noise_mags.append(np.linalg.norm(nt[int(i)]))
    mean_noise_mag = np.mean(noise_mags)
    reporter.log({'avg noise mag': float(mean_noise_mag)})

    gbest_dir, lbest_dir = global_best.theta - policy.flat_params, local_best.theta - policy.flat_params
    extra_scaled_noise = np.dot(extra_ranked, np.array(
        [(gbest_dir / np.linalg.norm(gbest_dir)) * mean_noise_mag,
         (lbest_dir / np.linalg.norm(lbest_dir)) * mean_noise_mag]))

    rews_ranked = CenteredRanker().rank(results[:, 0], results[:, 2], np.array(inds))
    scaled_ws = np.dot(rews_ranked, np.array(ws))  # only scaling w according the reward not the novelty

    steps = comm.allreduce(sum([tr.steps for tr in results_pos + results_neg]), op=MPI.SUM)
    gen_obstat.mpi_inc(comm)

    _approx_grad(ranked, ranker.n_fits_ranked, ranker.noise_inds, nt, policy.flat_params_w, optim, cfg, scaled_ws,
                 extra_scaled_noise)
    noiseless_result = fit_fn(policy.pheno(np.zeros(len(policy))), env, cfg.env.max_steps, rs)
    reporter.log_gen(ranker.fits, noiseless_result, policy, steps, time.time() - gen_start)

    idx_best = np.argmax(ranker.fits[:, 0])
    best_theta = policy.flat_params + \
                 ranker.noise_inds[idx_best] if idx_best < len(ranker.noise_inds) else -ranker.noise_inds[
        idx_best % len(ranker.noise_inds)]
    best = SavingResult(best_theta, ranker.fits[idx_best],
                        np.clip(policy.w + ws[idx_best % len(ranker.noise_inds)], 0, 1))
    return noiseless_result, gen_obstat, best


def _share_results(comm: MPI.Comm,
                   fits_pos: List[List[float]],
                   fits_neg: List[List[float]],
                   inds: List[int],
                   ws: List[float]) -> Tuple[ndarray, ndarray]:
    """share results and noise inds to all processes"""
    send_results = np.array([fp + fn + [i] for fp, fn, i in zip(fits_pos, fits_neg, inds)] * comm.size, dtype=np.float)
    results = np.empty(send_results.shape)
    comm.Alltoall(send_results, results)
    weight_noise = np.ravel(comm.alltoall([ws] * comm.size))

    objectives = len(fits_pos[0])
    return results.reshape((-1, 1 + 2 * objectives)), weight_noise  # flattening the process dim


def _approx_grad(ranked: ndarray, n: int, inds: ndarray, nt: NoiseTable, flat_params: ndarray, optim: Optimizer, cfg,
                 scaled_ws, extra_scaled_noise):
    """approximating gradient and update policy params"""
    grad = (scale_noise(ranked, inds, nt, cfg.general.batch_size) + extra_scaled_noise) / (n + 2)
    grad = np.concatenate((grad, np.array([scaled_ws])))
    optim.step(cfg.policy.l2coeff * flat_params - grad)
