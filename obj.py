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
from src.nn.nn import FeedForward, BaseNet
from src.nn.optimizers import Adam
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
    rs, my_seed, global_seed = utils.seed(comm, cfg.general.seed, env)
    all_seeds = comm.alltoall([my_seed] * comm.size)  # simply for saving the seeds used on each proc
    reporter.print(f'seeds:{all_seeds}')

    # initializing policy, optimizer, noise and env
    if 'load' in cfg.policy:
        policy: Policy = Policy.load(cfg.policy.load)
        nn: BaseNet = policy._module
    else:
        nn: BaseNet = FeedForward(cfg.policy.layer_sizes, torch.nn.Tanh(), env, cfg.policy.ac_std, cfg.policy.ob_clip)
        policy: Policy = Policy(nn, cfg, Adam)
    # optim: Optimizer = Adam(policy, cfg.policy.lr)

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
            rew, behv, obs, steps = gym_runner.run_model(model, env, cfg.env.max_steps, rs if use_ac_noise else None)
            rews[:len(rew)] += np.array(rew)

        rews /= max(1, cfg.general.eps_per_policy)
        return RewardResult(rews.tolist(), behv, obs if save_obs else np.array([np.zeros(env.observation_space.shape)]),
                            steps)

    time_since_best = 0
    noise_std_inc = 0.08
    for gen in range(cfg.general.gens):
        if cfg.general.mlflow: mlflow_reporter.set_active_run(0)
        reporter.start_gen()

        if cfg.noise.std_decay != 1:
            reporter.log({'noise std': policy.std})
        if cfg.policy.lr_decay != 1:
            reporter.log({'lr': policy.optim.lr})
        if cfg.policy.ac_std_decay != 1:
            reporter.log({'ac std': nn._action_std})

        tr, gen_obstat = es.step(cfg, comm, policy, nt, env, r_fn, rs, ranker, reporter)
        policy.update_obstat(gen_obstat)

        cfg.policy.ac_std = nn._action_std = nn._action_std * cfg.policy.ac_std_decay
        cfg.noise.std = policy.std = max(cfg.noise.std * cfg.noise.std_decay, cfg.noise.std_limit)
        cfg.policy.lr = policy.optim.lr = max(cfg.policy.lr * cfg.policy.lr_decay, cfg.policy.lr_limit)

        reporter.log({'obs recorded': policy.obstat.count})

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
