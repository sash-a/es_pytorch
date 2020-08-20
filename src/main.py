import random
from typing import List

import gym
import numpy as np
import torch
from mpi4py import MPI

import es.es_runner as es
from es.noisetable import NoiseTable
from es.optimizers import Adam, Optimizer
from es.policy import Policy
from utils import utils, gym_runner
from utils.nn import FullyConnected
from utils.novelty import novelty, update_archive
from utils.reporters import LoggerReporter

BEHAVIOUR = 'behaviour'
REWARD = 'reward'

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg = utils.load_config(utils.parse_args())

    # This is required for the moment, as parameter initialization needs to be deterministic across all processes
    assert cfg.policy.seed is not None
    torch.random.manual_seed(cfg.policy.seed)
    rs = np.random.RandomState()  # this must not be seeded, otherwise all procs will use the same random noise

    population: List[Policy] = [
        Policy(FullyConnected(15, 3, 256, 2, torch.nn.Tanh, cfg.policy), cfg.noise.std) for _ in
        range(cfg.general.n_policies)
    ]
    optims: List[Optimizer] = [Adam(policy, cfg.general.lr) for policy in population]
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.table_size, len(population[0]), cfg.noise.seed)
    env: gym.Env = gym.make(cfg.env.name)
    reporter = LoggerReporter(comm, cfg, cfg.general.name)

    archive = []
    policy_fits = []
    best_rew = 0


    def ns_fn(model: torch.nn.Module,
              e: gym.Env,
              max_steps: int,
              r: np.random.RandomState = None):
        rews, behv = gym_runner.run_model(model, e, max_steps, r)
        fit = novelty(np.array([behv]), archive, cfg.novelty.n)

        return fit, {BEHAVIOUR: behv, REWARD: sum(rews)}


    def r_fn(model: torch.nn.Module,
             e: gym.Env,
             max_steps: int,
             r: np.random.RandomState = None):
        rews, behv = gym_runner.run_model(model, e, max_steps, r)

        return sum(rews), {BEHAVIOUR: behv, REWARD: sum(rews)}
        # return behv[-1]


    for policy in population:
        _, info = r_fn(policy.pheno(np.zeros(len(policy))), env, cfg.env.max_steps, rs)
        archive.append(info[BEHAVIOUR])

    archive = np.array(archive)

    for behaviour in archive:
        policy_fits.append(novelty(np.array([behaviour]), archive, cfg.novelty.n))

    for gen in range(cfg.general.gens):
        idx = random.choices(list(range(cfg.general.n_policies)), weights=policy_fits, k=1)[0]
        idx = comm.scatter([idx] * comm.size)
        fit, info = es.step(cfg, comm, population[idx], optims[idx], nt, env, ns_fn, rs, reporter=reporter)

        archive = update_archive(comm, info[BEHAVIOUR], archive)
        policy_fits[idx] = max(policy_fits[idx], fit)
