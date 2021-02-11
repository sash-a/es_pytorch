import os

import gym
import mlflow
import numpy as np
import torch
from mpi4py import MPI

import es_pytorch.src.core.es as es
from es_pytorch.src.core.noisetable import NoiseTable
from es_pytorch.src.core.policy import Policy
from es_pytorch.src.gym import gym_runner
from es_pytorch.src.gym.training_result import TrainingResult, RewardResult
from es_pytorch.src.gym.unity import UnityGymWrapper
from es_pytorch.src.nn.nn import FullyConnected
from es_pytorch.src.nn.obstat import ObStat
from es_pytorch.src.nn.optimizers import Adam, Optimizer
from es_pytorch.src.utils import utils
from es_pytorch.src.utils.rankers import CenteredRanker
from es_pytorch.src.utils.reporters import MLFlowReporter, LoggerReporter, StdoutReporter, DefaultMpiReporterSet
from es_pytorch.src.utils.utils import generate_seed

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    full_name = f'{os.path.split(cfg.env.name)[1][:-7]}-{cfg.general.name}'
    env: gym.Env = UnityGymWrapper(cfg.env.name, comm.rank)

    mlflow_reporter = MLFlowReporter(comm, cfg) if cfg.general.mlflow else None
    if cfg.general.mlflow: mlflow_reporter.set_active_run(0)
    reporter = DefaultMpiReporterSet(comm, full_name,
                                     LoggerReporter(comm, full_name),
                                     StdoutReporter(comm),
                                     mlflow_reporter)

    # seeding; this must be done before creating the neural network so that params are deterministic across processes
    cfg.general.seed = (generate_seed(comm) if cfg.general.seed is None else cfg.general.seed)
    rs = utils.seed(comm, cfg.general.seed, env)

    # initializing obstat, policy, optimizer, noise and ranker
    obstat: ObStat = ObStat(env.observation_space.shape, 1e-2)  # eps to prevent dividing by zero at the beginning
    nn = FullyConnected(int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape)), 256, 2,
                        torch.nn.Tanh(), env, cfg.policy)
    policy: Policy = Policy(nn, cfg.noise.std)
    optim: Optimizer = Adam(policy, cfg.policy.lr)
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(policy), None, cfg.general.seed)
    ranker = CenteredRanker()


    def r_fn(model: torch.nn.Module) -> TrainingResult:
        save_obs = (rs.random() if rs is not None else np.random.random()) < cfg.policy.save_obs_chance
        rews, behv, obs, steps = gym_runner.run_model(model, env, cfg.env.max_steps, rs, save_obs)
        return RewardResult(rews, behv, obs, steps)


    assert cfg.general.policies_per_gen % comm.size == 0 and (cfg.general.policies_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.policies_per_gen / comm.size) / 2)
    for gen in range(cfg.general.gens):  # main loop
        reporter.start_gen()
        nn.set_ob_mean_std(obstat.mean, obstat.std)  # for normalizing the observation space

        gen_obstat = ObStat(env.observation_space.shape, 0)  # for normalizing the observation space
        pos_fits, neg_fits, inds, steps = es.test_params(comm, eps_per_proc, policy, nt, gen_obstat, r_fn, rs)
        obstat += gen_obstat  # adding the new observations to the global obstat
        ranker.rank(pos_fits, neg_fits, inds)
        es.approx_grad(ranker, nt, policy.flat_params, optim, cfg.general.batch_size, cfg.policy.l2coeff)

        reporter.log_gen(ranker.fits, r_fn(policy.pheno()), policy, steps)
        reporter.end_gen()

    mlflow.end_run()
