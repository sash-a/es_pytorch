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
from es.utils.rankers import CenteredRanker, EliteRanker
from es.utils.reporters import LoggerReporter, ReporterSet, StdoutReporter, MLFlowReporter
from es.utils.training_result import TrainingResult, RewardResult
from es.utils.utils import generate_seed

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD
    gym.logger.set_level(40)

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    mlflow_reporter = MLFlowReporter(comm, cfg_file, cfg) if cfg.general.mlflow else None
    reporter = ReporterSet(
        LoggerReporter(comm, cfg, cfg.general.name),
        StdoutReporter(comm),
        mlflow_reporter
    )

    env: gym.Env = gym.make(cfg.env.name)

    # seeding
    cfg.general.seed = (generate_seed(comm) if cfg.general.seed is None else cfg.general.seed)
    rs = utils.seed(comm, cfg.general.seed, env)
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
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(policy), reporter, cfg.general.seed)

    ranker = CenteredRanker()
    if 0 < cfg.experimental.elite < 1:
        ranker = EliteRanker(CenteredRanker(), cfg.experimental.elite)

    best_rew = -np.inf
    best_dist = -np.inf
    best_max_rew = -np.inf  # highest achieved in any gen


    def r_fn(model: torch.nn.Module) -> TrainingResult:
        save_obs = rs.random() < cfg.policy.save_obs_chance
        rews, behv, obs, steps = gym_runner.run_model(model, env, cfg.env.max_steps, rs, save_obs)
        return RewardResult(rews, behv, obs, steps)


    time_since_best = 0
    noise_std_inc = 0.08
    for gen in range(cfg.general.gens):
        if cfg.general.mlflow: mlflow_reporter.set_active_run(0)
        reporter.start_gen()

        if cfg.noise.std_decay != 1:
            reporter.log({'noise std': cfg.noise.std})
        if cfg.policy.lr_decay != 1:
            reporter.log({'lr': cfg.policy.lr})

        nn.set_ob_mean_std(obstat.mean, obstat.std)
        tr, gen_obstat = es.step(cfg, comm, policy, optim, nt, env, r_fn, rs, ranker, reporter)
        obstat += gen_obstat  # adding the new observations to the global obstat

        cfg.noise.std = policy.std = max(cfg.noise.std * cfg.noise.std_decay, cfg.noise.std_limit)
        cfg.policy.lr = optim.lr = max(cfg.policy.lr * cfg.policy.lr_decay, cfg.policy.lr_limit)

        reporter.log({'obs recorded': obstat.count})

        dist = np.linalg.norm(np.array(tr.positions[-3:-1]))
        rew = np.sum(tr.rewards)
        max_rew = np.max(ranker.fits[:, 0])

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
        best_max_rew = max(best_max_rew, max_rew)

        # Saving policy if it obtained a better reward or distance
        if save_policy and comm.rank == 0:
            policy.save(f'saved/{cfg.general.name}', str(gen))
            reporter.print(f'saving policy with rew:{rew:0.2f} and dist:{dist:0.2f}')

        reporter.end_gen()
    mlflow.end_run()  # in the case where mlflow is the reporter, just ending its run
