from functools import partial

import gym
import mlflow
import numpy as np
import torch
from mpi4py import MPI

import es.evo.es as es
from es.evo.noisetable import NoiseTable
from es.evo.policy import Policy
from es.nn.nn import FullyConnected
from es.nn.optimizers import Adam, Optimizer
from es.utils import utils, gym_runner
from es.utils.obstat import ObStat
from es.utils.reporters import LoggerReporter, ReporterSet, StdoutReporter, MLFlowReporter
from es.utils.training_result import TrainingResult, RewardResult, XDistResult
from es.utils.utils import moo_mean_rank, generate_seed, compute_centered_ranks

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD
    gym.logger.set_level(40)

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    reporter = ReporterSet(
        LoggerReporter(comm, cfg, cfg.general.name),
        StdoutReporter(comm),
        MLFlowReporter(comm, cfg_file, cfg)
    )

    env: gym.Env = gym.make(cfg.env.name)

    # seeding
    cfg.general.seed = (generate_seed(comm) if cfg.general.seed is None else cfg.general.seed)
    rs = np.random.RandomState(cfg.general.seed + 10000 * comm.rank)  # This seed must be different on each proc
    torch.random.manual_seed(cfg.general.seed)  # This seed must be the same on each proc for generating initial params
    env.seed(cfg.general.seed)
    reporter.print(f'seed:{cfg.general.seed}')

    # initializing policy, optimizer, noise and env
    obstat: ObStat = ObStat(env.observation_space.shape, 1e-2)  # eps to prevent dividing by zero at the beginning
    nn = FullyConnected(int(np.prod(env.observation_space.shape)),
                        int(np.prod(env.action_space.shape)),
                        256,
                        2,
                        torch.nn.Tanh(),
                        env,
                        cfg.policy)
    policy: Policy = Policy(nn, cfg.noise.std)
    optim: Optimizer = Adam(policy, cfg.policy.lr)
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.table_size, len(policy), reporter, cfg.general.seed)

    rank_fn = partial(moo_mean_rank, rank_fn=compute_centered_ranks)

    best_rew = -np.inf
    best_dist = -np.inf


    def r_fn(model: torch.nn.Module, e: gym.Env, max_steps: int, r: np.random.RandomState = None) -> TrainingResult:
        save_obs = (r.random() if r is not None else np.random.random()) < cfg.policy.save_obs_chance
        rews, behv, obs, steps = gym_runner.run_model(model, e, max_steps, r, save_obs)
        return RewardResult(rews, behv, obs, steps)


    def dist_fn(model: torch.nn.Module, e: gym.Env, max_steps: int, r: np.random.RandomState = None) -> TrainingResult:
        save_obs = (r.random() if r is not None else np.random.random()) < cfg.policy.save_obs_chance
        rews, behv, obs, steps = gym_runner.run_model(model, e, max_steps, r, save_obs)
        return XDistResult(rews, behv, obs, steps)


    time_since_best = 0
    noise_std_inc = 0.01
    for gen in range(cfg.general.gens):
        if cfg.noise.std_decay != 1:
            reporter.log({'noise std': cfg.noise.std})
        if cfg.policy.lr_decay != 1:
            reporter.log({'lr': cfg.policy.lr})

        nn.set_ob_mean_std(obstat.mean, obstat.std)
        tr, gen_obstat = es.step(cfg, comm, policy, optim, nt, env, r_fn, rs, rank_fn, reporter)
        obstat += gen_obstat  # adding the new observations to the global obstat

        cfg.noise.std = policy.std = max(cfg.noise.std * cfg.noise.std_decay, cfg.noise.std_limit)
        cfg.policy.lr = optim.lr = max(cfg.policy.lr * cfg.policy.lr_decay, cfg.policy.lr_limit)

        reporter.log({'obs recorded': obstat.count})

        dist = np.linalg.norm(np.array(tr.behaviour[-3:-1]))
        rew = np.sum(tr.rewards)

        # increasing noise std if policy is stuck
        time_since_best = 0 if rew > best_rew else time_since_best + 1
        if time_since_best > 15 and cfg.experimental.explore_with_large_noise:
            cfg.noise.std = policy.std = policy.std + noise_std_inc

        # Saving policy if it obtained a better reward or distance
        if (rew > best_rew or dist > best_dist) and comm.rank == 0:
            best_rew = max(rew, best_rew)
            best_dist = max(dist, best_dist)
            policy.save(f'saved/{cfg.general.name}', str(gen))
            reporter.print(f'saving policy with rew:{rew:0.2f} and dist:{dist:0.2f}')

    mlflow.end_run()  # in the case where mlflow is the reporter, just ending its run
