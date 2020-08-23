from functools import partial

import gym
import numpy as np
import torch
from mpi4py import MPI

import es.es_runner as es
from es.noisetable import NoiseTable
from es.optimizers import Adam, Optimizer
from es.policy import Policy
from utils import utils, gym_runner
# noinspection PyUnresolvedReferences
from utils.TrainingResult import TrainingResult, DistResult, XDistResult, RewardResult
from utils.nn import FullyConnected
from utils.reporters import LoggerReporter
from utils.utils import moo_mean_rank, compute_centered_ranks

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg = utils.load_config(utils.parse_args())

    # This is required for the moment, as parameter initialization needs to be deterministic across all processes
    assert cfg.policy.seed is not None
    torch.random.manual_seed(cfg.policy.seed)
    rs = np.random.RandomState()  # this must not be seeded, otherwise all procs will use the same random noise

    # initializing policy, optimizer, noise and env
    policy: Policy = Policy(FullyConnected(15, 3, 256, 2, torch.nn.Tanh, cfg.policy), cfg.noise.std)
    optim: Optimizer = Adam(policy, cfg.general.lr)
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.table_size, len(policy), cfg.noise.seed)
    env: gym.Env = gym.make(cfg.env.name)
    reporter = LoggerReporter(comm, cfg, cfg.general.name)

    rank_fn = partial(moo_mean_rank, rank_fn=compute_centered_ranks)


    def r_fn(model: torch.nn.Module, e: gym.Env, max_steps: int, r: np.random.RandomState = None) -> TrainingResult:
        rews, behv = gym_runner.run_model(model, e, max_steps, r)
        return DistResult(rews, behv)
        # return XDistResult(rews, behv)
        # return RewardResult(rews, behv)


    for gen in range(cfg.general.gens):
        tr = es.step(cfg, comm, policy, optim, nt, env, r_fn, rs, rank_fn, reporter)
