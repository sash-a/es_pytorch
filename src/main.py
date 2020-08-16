import gym
import numpy as np
# import mkl
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
    # mkl.set_num_threads(1)  # https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66

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

    run(cfg, comm, policy, optim, nt, env, rs, utils.compute_centered_ranks, gym_runner.model_dist, reporter)
