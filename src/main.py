import gym
import numpy as np
import torch
from mpi4py import MPI

from es.es_runner import run
from es.noisetable import NoiseTable
from es.optimizers import Optimizer, Adam
from es.policy import Policy
from utils import gym_runner, utils
from utils.nn import FullyConnected
from utils.reporters import LoggerReporter

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg = utils.load_config(utils.parse_args())

    # This is required for the moment, as parameter initialization needs to be deterministic across all processes
    assert cfg.policy.seed is not None
    torch.random.manual_seed(cfg.policy.seed)
    rs = np.random.RandomState()  # this must not be seeded, otherwise all procs will use the same random noise

    policy: Policy = Policy(FullyConnected(22, 6, 256, 2, torch.nn.Tanh, cfg.policy), cfg.noise.std)
    optim: Optimizer = Adam(policy, cfg.general.lr)
    nt: NoiseTable = NoiseTable.create_shared_noisetable(comm, cfg.noise.table_size, len(policy), cfg.noise.seed)
    env: gym.Env = gym.make(cfg.env.name)
    reporter = LoggerReporter(comm, cfg, cfg.general.name)


    def fit_fn(model: torch.nn.Module,
               e: gym.Env,
               max_steps: int,
               r: np.random.RandomState = None,
               episodes: int = 1,
               render: bool = False):
        rew, dist = gym_runner.run_model(model, e, max_steps, r, episodes, render)
        return rew[0] + 4 * dist[0]


    run(cfg, comm, policy, optim, nt, env, rs, utils.compute_centered_ranks, fit_fn, reporter)
