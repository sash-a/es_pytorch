import gym
import torch
from mpi4py import MPI

from utils import gym_runner, utils
from es.es_runner import run
from utils.utils import parse_args, load_config
from utils.nn_structures import FullyConnected
from es.noisetable import NoiseTable
from es.optimizers import ES
from es.policy import Policy
from utils.reporter import Reporter

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg = load_config(parse_args())

    # This is required for the moment, as parameter initialization needs to be deterministic across all processes
    torch.random.manual_seed(cfg.seed)
    policy: Policy = Policy(FullyConnected(15, 3, 256, 2, torch.nn.Tanh), cfg.noise_stdev)
    optim: ES = ES(policy, cfg.lr, cfg.eps_per_gen)
    nt: NoiseTable = NoiseTable.create_shared_noisetable(comm, cfg.table_size, len(policy), cfg.seed)
    env: gym.Env = gym.make(cfg.env_name)
    reporter = Reporter()

    run(cfg, comm, policy, optim, nt, env, utils.percent_rank, gym_runner.run_model, reporter)
