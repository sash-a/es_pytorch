import gym
# import mkl
import torch
import numpy as np
from mpi4py import MPI

from es.es_runner import run
from es.noisetable import NoiseTable
from es.optimizers import ES, Optimizer, Adam
from es.policy import Policy
from utils import gym_runner, utils
from utils.nn import FullyConnected
from utils.reporters import StdoutReporter, LoggerReporter

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD
    # mkl.set_num_threads(1)  # https://towardsdatascience.com/paper-repro-deep-neuroevolution-756871e00a66

    cfg = utils.load_config(utils.parse_args())

    # This is required for the moment, as parameter initialization needs to be deterministic across all processes
    torch.random.manual_seed(cfg.general.seed)
    rs = np.random.RandomState()

    policy: Policy = Policy(FullyConnected(26, 6, 256, 2, torch.nn.Tanh, cfg.policy), cfg.noise.std)
    optim: Optimizer = Adam(policy, cfg.general.lr)
    nt: NoiseTable = NoiseTable.create_shared_noisetable(comm, cfg.noise.table_size, len(policy), cfg.general.seed)
    env: gym.Env = gym.make(cfg.env.name)
    reporter = LoggerReporter(comm, cfg)

    run(cfg, comm, policy, optim, nt, env, rs, utils.compute_centered_ranks, gym_runner.run_model, reporter)
