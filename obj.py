from functools import partial

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
# noinspection PyUnresolvedReferences
from es.utils.ObStat import ObStat
from es.utils.TrainingResult import TrainingResult, RewardResult
from es.utils.reporters import LoggerReporter, ReporterSet, StdoutReporter, MLFlowReporter
from es.utils.utils import moo_mean_rank, compute_centered_ranks

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    # This is required for the moment, as parameter initialization needs to be deterministic across all processes
    assert cfg.policy.seed is not None
    torch.random.manual_seed(cfg.policy.seed)
    rs = np.random.RandomState()  # this must not be seeded, otherwise all procs will use the same random noise

    # initializing policy, optimizer, noise and env
    env: gym.Env = gym.make(cfg.env.name)
    obstat: ObStat = ObStat(env.observation_space.shape, 1e-2)  # eps to prevent dividing by zero at the beginning
    policy: Policy = Policy(
        FullyConnected(np.prod(env.observation_space.shape),
                       np.prod(env.action_space.shape),
                       256,
                       2,
                       torch.nn.Tanh,
                       obstat,
                       cfg.policy),
        cfg.noise.std)
    optim: Optimizer = Adam(policy, cfg.general.lr)
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.table_size, len(policy), cfg.noise.seed)

    reporter = ReporterSet(
        LoggerReporter(comm, cfg, cfg.general.name),
        StdoutReporter(comm),
        MLFlowReporter(comm, cfg_file, cfg)
    )

    rank_fn = partial(moo_mean_rank, rank_fn=compute_centered_ranks)


    def r_fn(model: torch.nn.Module, e: gym.Env, max_steps: int, r: np.random.RandomState = None) -> TrainingResult:
        save_obs = (r.random() if r is not None else np.random.random()) < cfg.policy.save_obs_chance
        rews, behv, obs = gym_runner.run_model(model, e, max_steps, r, save_obs)
        # return DistResult(rews, behv, obs)
        # return XDistResult(rews, behv, obs)
        return RewardResult(rews, behv, obs)


    for gen in range(cfg.general.gens):
        tr = es.step(cfg, comm, policy, optim, nt, env, r_fn, rs, rank_fn, obstat, reporter)
