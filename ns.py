import random
from functools import partial
from typing import List

import gym
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
from es.utils.reporters import LoggerReporter
from es.utils.training_result import TrainingResult, NSResult
from es.utils.utils import compute_centered_ranks, moo_mean_rank

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg = utils.load_config(utils.parse_args())

    # This is required for the moment, as parameter initialization needs to be deterministic across all processes
    assert cfg.policy.seed is not None
    torch.random.manual_seed(cfg.policy.seed)
    rs = np.random.RandomState()  # this must not be seeded, otherwise all procs will use the same random noise

    # initializing population, optimizers, noise and env
    env: gym.Env = gym.make(cfg.env.name)
    population: List[Policy] = [
        Policy(FullyConnected(env.observation_space.shape[0], env.action_space.shape[0], 256, 2, torch.nn.Tanh,
                              cfg.policy), cfg.noise.std) for _ in range(cfg.general.n_policies)
    ]
    optims: List[Optimizer] = [Adam(policy, cfg.policy.lr) for policy in population]
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.table_size, len(population[0]), cfg.noise.seed)
    reporter = LoggerReporter(comm, cfg, cfg.general.name)

    archive = None
    policies_best_fit = []

    TR = NSResult
    rank_fn = partial(moo_mean_rank, rank_fn=compute_centered_ranks)

    def ns_fn(model: torch.nn.Module, e: gym.Env, max_steps: int, r: np.random.RandomState = None) -> TrainingResult:
        rews, behv = gym_runner.run_model(model, e, max_steps, r)
        return TR(rews, behv, archive, cfg.novelty.k)


    # initializing the archive
    initial_results = []
    for policy in population:
        rews, behaviour = gym_runner.run_model(policy.pheno(np.zeros(len(policy))), env, cfg.env.max_steps, rs)
        archive = update_archive(comm, behaviour[-3:-1], archive)
        initial_results.append(TR(rews, behaviour, archive, cfg.novelty.k))

    for initial_result in initial_results:
        initial_result = comm.scatter([initial_result.result[0]] * comm.size)
        policies_best_fit.append(initial_result)

    for gen in range(cfg.general.gens):
        # picking the policy from the population
        idx = random.choices(list(range(len(policies_best_fit))), weights=policies_best_fit, k=1)[0]
        idx = comm.scatter([idx] * comm.size)
        # running es
        tr = es.step(cfg, comm, population[idx], optims[idx], nt, env, ns_fn, rs, rank_fn, reporter)
        tr = comm.scatter([tr] * comm.size)
        # adding new behaviour and sharing archive
        archive = update_archive(comm, tr.behaviour[-3:-1], archive)
