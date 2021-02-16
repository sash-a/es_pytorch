from os import path

import gym
import mlflow
import numpy as np
import torch
from mpi4py import MPI

import src.core.es as es
from src.core.noisetable import NoiseTable
from src.core.policy import Policy
from src.gym import gym_runner
from src.gym.training_result import TrainingResult, RewardResult
from src.nn.nn import FullyConnected
from src.nn.obstat import ObStat
from src.nn.optimizers import Adam, Optimizer
from src.utils import utils
from src.utils.rankers import CenteredRanker, EliteRanker
from src.utils.reporters import LoggerReporter, ReporterSet, StdoutReporter, MLFlowReporter


def main(cfg):
    comm: MPI.Comm = MPI.COMM_WORLD

    full_name = f'{cfg.env.name}-{cfg.general.name}'
    mlflow_reporter = MLFlowReporter(comm, cfg) if cfg.general.mlflow else None
    reporter = ReporterSet(
        LoggerReporter(comm, full_name),
        StdoutReporter(comm),
        mlflow_reporter
    )

    env: gym.Env = gym.make(cfg.env.name)

    # seeding
    cfg.general.seed = (utils.generate_seed(comm) if cfg.general.seed is None else cfg.general.seed)
    rs = utils.seed(comm, cfg.general.seed, env)
    reporter.print(f'seed:{cfg.general.seed}')

    # initializing policy, optimizer, noise and env
    obstat: ObStat = ObStat(env.observation_space.shape, 1e-2)  # eps to prevent dividing by zero at the beginning
    nn = FullyConnected(int(np.prod(env.observation_space.shape)),
                        int(np.prod(env.action_space.shape)),
                        256, 2, torch.nn.Tanh(), env, cfg.policy)
    policy: Policy = Policy(nn, cfg.noise.std)

    optim: Optimizer = Adam(policy, cfg.policy.lr)
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(policy), reporter, cfg.general.seed)

    ranker = CenteredRanker()
    if 0 < cfg.experimental.elite < 1:
        ranker = EliteRanker(CenteredRanker(), cfg.experimental.elite)

    best_rew = -np.inf
    best_dist = -np.inf
    best_max_rew = -np.inf  # highest achieved in any gen

    def r_fn(model: torch.nn.Module, use_ac_noise=True) -> TrainingResult:
        save_obs = rs.random() < cfg.policy.save_obs_chance
        rews = np.zeros(cfg.env.max_steps)
        for _ in range(max(1, cfg.general.eps_per_policy)):
            rew, behv, obs, steps = gym_runner.run_model(model, env, cfg.env.max_steps,
                                                         rs if use_ac_noise else None, save_obs)
            rews[:len(rew)] += np.array(rew)

        rews /= max(1, cfg.general.eps_per_policy)
        return RewardResult(rews.tolist(), behv, obs, steps)

    time_since_best = 0
    noise_std_inc = 0.08
    for gen in range(cfg.general.gens):
        if cfg.general.mlflow: mlflow_reporter.set_active_run(0)
        reporter.start_gen()

        if cfg.noise.std_decay != 1:
            reporter.log({'noise std': policy.std})
        if cfg.policy.lr_decay != 1:
            reporter.log({'lr': optim.lr})
        if cfg.policy.ac_std_decay != 1:
            reporter.log({'ac std': nn._action_std})

        nn.set_ob_mean_std(obstat.mean, obstat.std)
        tr, gen_obstat = es.step(cfg, comm, policy, optim, nt, env, r_fn, rs, ranker, reporter)
        obstat += gen_obstat  # adding the new observations to the global obstat

        cfg.policy.ac_std = nn._action_std = nn._action_std * cfg.policy.ac_std_decay
        cfg.noise.std = policy.std = max(cfg.noise.std * cfg.noise.std_decay, cfg.noise.std_limit)
        cfg.policy.lr = optim.lr = max(cfg.policy.lr * cfg.policy.lr_decay, cfg.policy.lr_limit)

        reporter.log({'obs recorded': obstat.count})

        dist = np.linalg.norm(np.array(tr.positions[-3:-1]))
        rew = np.sum(tr.rewards)
        max_rew_ind = np.argmax(ranker.fits[:, 0])
        max_rew = ranker.fits[:, 0][max_rew_ind]

        time_since_best = 0 if max_rew > best_max_rew else time_since_best + 1
        reporter.log({'time since best': time_since_best})
        # increasing noise std if policy is stuck
        if time_since_best > cfg.experimental.max_time_since_best and cfg.experimental.explore_with_large_noise:
            cfg.noise.std = policy.std = policy.std + noise_std_inc

        if 0 < cfg.experimental.elite < 1:  # using elite extension
            if time_since_best > cfg.experimental.max_time_since_best and cfg.experimental.elite < 1:
                ranker.elite_percent = cfg.experimental.elite
            if time_since_best == 0:
                ranker.elite_percent = 1
            reporter.print(f'elite percent: {ranker.elite_percent}')

        save_policy = (rew > best_rew or dist > best_dist)
        best_rew = max(rew, best_rew)
        best_dist = max(dist, best_dist)

        # Saving policy if it obtained a better reward or distance
        if save_policy and comm.rank == 0:
            policy.save(path.join('saved', full_name, 'weights'), str(gen))
            reporter.print(f'saving policy with rew:{rew:0.2f} and dist:{dist:0.2f}')

        # Saving max rew if it obtained best ever rew
        if max_rew > best_max_rew and comm.rank == 0:
            best_max_rew = max_rew
            coeff = 1 if max_rew_ind < ranker.n_fits_ranked // 2 else -1  # checking if pos or neg noise ind used
            torch.save(policy.pheno(coeff * ranker.noise_inds[max_rew_ind % (ranker.n_fits_ranked // 2)]),
                       path.join('saved', full_name, 'weights', f'gen{gen}-rew{best_max_rew:0.0f}.pt'))
            reporter.print(f'saving max policy with rew:{best_max_rew:0.2f}')

        reporter.end_gen()
    mlflow.end_run()  # in the case where mlflow is the reporter, just ending its run


if __name__ == '__main__':
    gym.logger.set_level(40)

    config_file = utils.parse_args()
    config = utils.load_config(config_file)

    main(config)
