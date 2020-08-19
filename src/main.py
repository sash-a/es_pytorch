from typing import Iterable

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
from utils.reporters import LoggerReporter
from utils.utils import novelty


def share_archive(comm: MPI.Comm, local_archive) -> Iterable:
    local_archive = np.array(local_archive)
    sizes = comm.alltoall([local_archive.size] * comm.size)
    send = local_archive * comm.size
    recv = np.zeros(sum(sizes), dtype=local_archive.dtype)
    comm.Alltoallv(send, (recv, sizes))
    return recv


if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg = utils.load_config(utils.parse_args())

    # This is required for the moment, as parameter initialization needs to be deterministic across all processes
    assert cfg.policy.seed is not None
    torch.random.manual_seed(cfg.policy.seed)
    rs = np.random.RandomState()  # this must not be seeded, otherwise all procs will use the same random noise

    policy: Policy = Policy(FullyConnected(15, 3, 256, 2, torch.nn.Tanh, cfg.policy), cfg.noise.std)
    optim: Optimizer = Adam(policy, cfg.general.lr)
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.table_size, len(policy), cfg.noise.seed)
    env: gym.Env = gym.make(cfg.env.name)
    reporter = LoggerReporter(comm, cfg, cfg.general.name)

    archive = [[0] * int(3 * cfg.env.max_steps / 10) for _ in range(2)]
    new_archive, behaviours = [], []


    def novelty_fit_fn(model: torch.nn.Module,
                       e: gym.Env,
                       max_steps: int,
                       r: np.random.RandomState = None):
        rews, behv = gym_runner.run_model(model, e, max_steps, r, False)
        others = np.array(archive + new_archive + behaviours)  # grouping all other indvs
        np_behv = np.array([behv], dtype=np.float64)
        fit = novelty(np_behv, others, cfg.novelty.n)

        behaviours.append(behv)
        if fit > cfg.novelty.thresh:
            new_archive.append(behv)

        return fit, {'dist': str(behv[-1]), 'rew': str(sum(rews))}


    def obj_fit_fn(model: torch.nn.Module,
                   e: gym.Env,
                   max_steps: int,
                   r: np.random.RandomState = None):
        rews, behv = gym_runner.run_model(model, e, max_steps, r, False)

        return sum(rews)
        # return behv[-1]


    for gen in range(cfg.general.gens):
        es.step(cfg, comm, policy, optim, nt, env, novelty_fit_fn, rs, utils.compute_centered_ranks, reporter)
        new_archive = share_archive(comm, new_archive)
        archive += new_archive
        behaviours, new_archive = [], []
