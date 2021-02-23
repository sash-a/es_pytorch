import os
import random
from os import path
from typing import List, Callable, Optional, Tuple

import gym
import mlflow
import numpy as np
import torch
from mpi4py import MPI
from munch import Munch

import src.core.es as es
from src.core.noisetable import NoiseTable
from src.core.policy import Policy
from src.gym import gym_runner
from src.gym.training_result import NSRResult, NSResult
from src.nn.nn import FeedForward
from src.nn.optimizers import Adam
from src.utils import utils
from src.utils.novelty import update_archive, novelty
from src.utils.rankers import CenteredRanker, MultiObjectiveRanker
from src.utils.reporters import LoggerReporter, StdoutReporter, MLFlowReporter, DefaultMpiReporterSet


def mean_behv(policy: Policy, r_fn: Callable[[torch.nn.Module], NSResult], rollouts: int):
    behvs = [r_fn(policy.pheno(np.zeros(len(policy)))).behaviour for _ in range(rollouts)]
    return np.mean(behvs, axis=0)


def init_archive(comm, cfg, pop: List[Policy], fn) -> Tuple[np.ndarray, List[float]]:
    """initializing the archive and policy weighting"""
    archive = None
    policies_novelties = []

    for policy in pop:
        b = None  # behaviour
        if comm.rank == 0:
            b = mean_behv(policy, fn, cfg.novelty.rollouts)
        archive = update_archive(comm, b, archive)
        b = archive[-1]
        nov = max(1e-2, novelty(b, archive, cfg.novelty.k))
        policies_novelties.append(nov)

    return archive, policies_novelties


def nsra(cfg: Munch, reward: float, obj_w: float, best_reward: float, time_since_best_reward: int) -> Tuple[float, float, float]:
    """
    Updates the weighting for NSRA-ES

    :returns Tuple[objective weighting, best reward, time since best]
    """
    if reward > best_reward:
        return min(1, obj_w + cfg.nsr.weight_delta), reward, 0
    else:
        time_since_best_reward += 1

        if time_since_best_reward > cfg.nsr.max_time_since_best:
            obj_w = max(0, obj_w - cfg.nsr.weight_delta)
            time_since_best_reward = 0

        return obj_w, best_reward, time_since_best_reward


def main(cfg: Munch):
    full_name = f'{cfg.env.name}-{cfg.general.name}'
    comm: MPI.Comm = MPI.COMM_WORLD
    env: gym.Env = gym.make(cfg.env.name)

    mlflow_reporter = MLFlowReporter(comm, cfg) if cfg.general.mlflow else None
    reporter = DefaultMpiReporterSet(comm, full_name,
                                     LoggerReporter(comm, full_name),
                                     StdoutReporter(comm),
                                     mlflow_reporter)

    # seeding
    rs, my_seed, global_seed = utils.seed(comm, cfg.general.seed, env)
    all_seeds = comm.alltoall([my_seed] * comm.size)  # simply for saving the seeds used on each proc
    reporter.print(f'seeds:{all_seeds}')

    if cfg.nsr.adaptive:
        reporter.print("NSRA")
    elif cfg.nsr.progressive:
        reporter.print("P-NSRA")

    archive: Optional[np.ndarray] = None

    def ns_fn(model: torch.nn.Module, use_ac_noise=True) -> NSRResult:
        """Reward function"""
        save_obs = rs.random() < cfg.policy.save_obs_chance
        rews, behv, obs, steps = gym_runner.run_model(model, env, cfg.env.max_steps, rs if use_ac_noise else None)
        return NSRResult(rews, behv[-3:], obs if save_obs else np.array([np.zeros(env.observation_space.shape)]), steps,
                         archive, cfg.novelty.k)

    # init population
    population = []
    nns = []
    for _ in range(cfg.general.n_policies):
        nns.append(FeedForward(cfg.policy.layer_sizes, torch.nn.Tanh(), env, cfg.policy.ac_std, cfg.policy.ob_clip))
        population.append(Policy(nns[-1], cfg.noise.std, Adam(len(Policy.get_flat(nns[-1])), cfg.policy.lr)))
    # init optimizer and noise table
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(population[0]), reporter, cfg.general.seed)
    policies_best_rewards = [-np.inf] * cfg.general.n_policies
    time_since_best = [0 for _ in range(cfg.general.n_policies)]  # TODO should this be per individual?
    obj_weight = [cfg.nsr.initial_w for _ in range(cfg.general.n_policies)]

    best_rew = -np.inf
    best_dist = -np.inf

    archive, policies_novelties = init_archive(comm, cfg, population, ns_fn)

    for gen in range(cfg.general.gens):  # main loop
        # picking the policy from the population
        idx = random.choices(list(range(len(policies_novelties))), weights=policies_novelties, k=1)[0]
        if cfg.nsr.progressive: idx = gen % cfg.general.n_policies
        idx = comm.scatter([idx] * comm.size)
        ranker = MultiObjectiveRanker(CenteredRanker(), obj_weight[idx])
        # reporting
        if cfg.general.mlflow: mlflow_reporter.set_active_run(idx)
        reporter.start_gen()
        reporter.log({'idx': idx})
        reporter.log({'w': obj_weight[idx]})
        reporter.log({'time since best': time_since_best[idx]})
        # running es
        tr, gen_obstat = es.step(cfg, comm, population[idx], nt, env, ns_fn, rs, ranker, reporter)
        for policy in population:
            policy.update_obstat(gen_obstat)  # shared obstat

        tr = comm.scatter([tr] * comm.size)  # sharing result
        # updating the weighting for choosing the next policy to be evaluated
        behv = comm.scatter([mean_behv(population[idx], ns_fn, cfg.novelty.rollouts)] * comm.size)
        nov = comm.scatter([novelty(behv, archive, cfg.novelty.k)] * comm.size)
        archive = update_archive(comm, behv, archive)  # adding new behaviour and sharing archive
        policies_novelties[idx] = nov

        dist = np.linalg.norm(np.array(tr.positions[-3:-1]))
        rew = tr.reward

        if cfg.nsr.adaptive:
            obj_weight[idx], policies_best_rewards[idx], time_since_best[idx] = nsra(cfg, rew, obj_weight[idx],
                                                                                     policies_best_rewards[idx],
                                                                                     time_since_best[idx])
        elif cfg.nsr.progressive:
            obj_weight[idx] = 1 if gen > cfg.nsr.end_progression_gen else gen / cfg.nsr.end_progression_gen

        # Saving policy if it obtained a better reward or distance
        if (rew > best_rew or dist > best_dist) and comm.rank == 0:
            best_rew = max(rew, best_rew)
            best_dist = max(dist, best_dist)

            # Only need to save the archive, policy is saved by DefaultMpiReportedSet
            archive_path = path.join('saved', full_name, 'archives')
            if not path.exists(archive_path):
                os.makedirs(archive_path)
            np.save(path.join(archive_path, f'{gen}.np'), archive)

        reporter.end_gen()

    mlflow.end_run()  # ending the outer mlflow run


if __name__ == '__main__':
    gym.logger.set_level(40)

    config_file = utils.parse_args()
    config = utils.load_config(config_file)

    main(config)
