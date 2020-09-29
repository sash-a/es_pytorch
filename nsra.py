import random
from functools import partial
from typing import List

import gym
import mlflow
import numpy as np
import torch
from mpi4py import MPI

import es.evo.es as es
from es.evo.noisetable import NoiseTable
from es.evo.policy import Policy
from es.nn.nn import FullyConnected
from es.nn.optimizers import Adam, Optimizer
from es.utils import utils, gym_runner
from es.utils.novelty import update_archive
from es.utils.obstat import ObStat
from es.utils.reporters import LoggerReporter, ReporterSet, StdoutReporter, MLFlowReporter
from es.utils.training_result import TrainingResult, NSRResult
from es.utils.utils import compute_centered_ranks, moo_weighted_rank, generate_seed

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD
    gym.logger.set_level(40)

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    cfg.general.seed = (generate_seed(comm) if cfg.general.seed is None else cfg.general.seed)
    torch.random.manual_seed(cfg.general.seed)
    rs = np.random.RandomState(cfg.general.seed + 10000 * comm.rank)

    reporter = ReporterSet(
        LoggerReporter(comm, cfg, cfg.general.name),
        StdoutReporter(comm),
        MLFlowReporter(comm, cfg_file, cfg)
    )
    reporter.print(f'seed:{cfg.general.seed}')

    # initializing population, optimizers, noise and env
    env: gym.Env = gym.make(cfg.env.name)

    in_size, out_size = np.prod(env.observation_space.shape), np.prod(env.action_space.shape)
    population = []
    nns = []
    for _ in range(cfg.general.n_policies):
        nns.append(FullyConnected(in_size, out_size, 256, 2, torch.nn.Tanh, env, cfg.policy))
        population.append(Policy(nns[-1], cfg.noise.std))

    optims: List[Optimizer] = [Adam(policy, cfg.policy.lr) for policy in population]
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.table_size, len(population[0]), reporter,
                                              cfg.general.seed)

    time_since_best = [0 for _ in range(cfg.general.n_policies)]
    obj_weight = [1. for _ in range(cfg.general.n_policies)]  # (1 - obj_weight[i]) is the novelty weighting

    archive = None
    policies_best_fit = []

    obstat: ObStat = ObStat(env.observation_space.shape, 1e-2)  # eps to prevent dividing by zero at the beginning
    best_rew = -np.inf
    best_dist = -np.inf


    def ns_fn(model: torch.nn.Module, e: gym.Env, max_steps: int, r: np.random.RandomState = None) -> TrainingResult:
        rews, behv, obs, steps = gym_runner.run_model(model, e, max_steps, r)
        return NSRResult(rews, behv, obs, steps, archive, cfg.novelty.k)


    # initializing the archive
    initial_results = []
    for policy in population:
        rews, behv, obs, steps = gym_runner.run_model(policy.pheno(np.zeros(len(policy))), env, cfg.env.max_steps, rs)
        archive = update_archive(comm, behv[-3:-1], archive)
        initial_results.append(NSRResult(rews, behv, obs, steps, archive, cfg.novelty.k))

    for initial_result in initial_results:
        initial_result = comm.scatter([max(1e-3, initial_result.result[0])] * comm.size)  # weights must be positive
        policies_best_fit.append(initial_result)

    for gen in range(cfg.general.gens):
        # picking the policy from the population
        idx = random.choices(list(range(len(policies_best_fit))), weights=policies_best_fit, k=1)[0]
        idx = comm.scatter([idx] * comm.size)
        nns[idx].set_ob_mean_std(obstat.mean, obstat.std)
        rank_fn = partial(moo_weighted_rank, w=obj_weight[idx], rank_fn=compute_centered_ranks)
        # running es
        tr, gen_obstat = es.step(cfg, comm, population[idx], optims[idx], nt, env, ns_fn, rs, rank_fn, reporter)
        # sharing result and obstat
        tr = comm.scatter([tr] * comm.size)
        gen_obstat.mpi_inc(comm)
        obstat += gen_obstat

        archive = update_archive(comm, tr.behaviour[-3:-1], archive)  # adding new behaviour and sharing archive

        reporter.print(f'idx:{idx}')
        reporter.print(f'w:{obj_weight[idx]}')
        reporter.print(f'time since best:{time_since_best[idx]}')

        # updating the weighting for NSRA-ES
        fit = tr.result[0]  # tr.result[1] is novelty
        if fit > policies_best_fit[idx]:
            policies_best_fit[idx] = fit
            time_since_best[idx] = 0
            obj_weight[idx] = min(1, obj_weight[idx] + cfg.nsra.weight_delta)
        else:
            time_since_best[idx] += 1

        if time_since_best[idx] > cfg.nsra.max_time_since_best:
            obj_weight[idx] = max(0, obj_weight[idx] - cfg.nsra.weight_delta)
            time_since_best[idx] = 0

        # Saving policy if it obtained a better reward or distance
        dist = np.linalg.norm(np.array(tr.behaviour[-3:-1]))
        rew = np.sum(tr.rewards)
        if (rew > best_rew or dist > best_dist) and comm.rank == 0:
            best_rew = max(rew, best_rew)
            best_dist = max(dist, best_dist)
            population[idx].save(f'saved/{cfg.general.name}', str(gen))
            reporter.print(f'saving policy with rew:{rew:0.2f} and dist:{dist:0.2f}')

    mlflow.end_run()  # in the case where mlflow is the reporter, ending its run
