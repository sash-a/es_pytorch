# SNER: Start Novelty search End Reward

from typing import List, Callable, Optional

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
from es.utils.novelty import update_archive, novelty
from es.utils.obstat import ObStat
from es.utils.rankers import CenteredRanker
from es.utils.reporters import LoggerReporter, ReporterSet, StdoutReporter, MLFlowReporter
from es.utils.training_result import NSResult, RewardResult
from es.utils.utils import generate_seed


def mean_behv(policy: Policy, fn: Callable[[torch.nn.Module], NSResult], rollouts: int):
    behvs = [fn(policy.pheno(np.zeros(len(policy)))).behaviour for _ in range(rollouts)]
    return np.mean(behvs, axis=0)


def exploring(generation: int):
    return generation <= cfg.sner.n_explorations


if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD
    gym.logger.set_level(40)

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    env: gym.Env = gym.make(cfg.env.name)

    # seeding
    cfg.general.seed = (generate_seed(comm) if cfg.general.seed is None else cfg.general.seed)
    rs = utils.seed(comm, cfg.general.seed, env)

    mlflow_reporter = MLFlowReporter(comm, cfg_file, cfg) if cfg.general.mlflow else None
    reporter = ReporterSet(
        LoggerReporter(comm, cfg, cfg.general.name),
        StdoutReporter(comm),
        mlflow_reporter
    )
    reporter.print(f'seed:{cfg.general.seed}')

    # init population
    in_size, out_size = np.prod(env.observation_space.shape), np.prod(env.action_space.shape)
    population = []
    for _ in range(cfg.general.n_policies):
        population.append(Policy(FullyConnected(int(in_size), int(out_size), 256, 2, torch.nn.Tanh(), env, cfg.policy),
                                 cfg.noise.std))
    # init optimizer and noise table
    optims: List[Optimizer] = [Adam(policy, cfg.policy.lr) for policy in population]
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(population[0]), reporter, cfg.general.seed)

    obstat: ObStat = ObStat(env.observation_space.shape, 1e-2)  # eps to prevent dividing by zero at the beginning

    archive: Optional[np.ndarray] = None
    policies_novelties = []
    policies_best_rewards = [-np.inf] * cfg.general.n_policies
    best_policies_params = []

    best_rew = -np.inf
    best_dist = -np.inf


    def ns_fn(model: torch.nn.Module) -> NSResult:
        rews, behv, obs, steps = gym_runner.run_model(model, env, cfg.env.max_steps, rs)
        return NSResult(rews, behv, obs, steps, archive, cfg.novelty.k)


    def r_fn(model: torch.nn.Module) -> RewardResult:
        rews, behv, obs, steps = gym_runner.run_model(model, env, cfg.env.max_steps, rs)
        return RewardResult(rews, behv, obs, steps)


    eval_fn = ns_fn  # the currently used evaluation function

    # initializing the archive and policy weighting
    for policy in population:
        behv = None
        nov = None
        if comm.rank == 0:
            behv = mean_behv(policy, ns_fn, cfg.novelty.rollouts)
        archive = update_archive(comm, behv, archive)
        behv = archive[-1]
        nov = max(1e-2, novelty(behv, archive, cfg.novelty.k))
        policies_novelties.append(nov)

    for gen in range(cfg.general.gens):  # main loop
        # picking the policy from the population
        idx = gen % len(population)  # TODO test novelty being the deciding factor for which individual is evaluated
        population[idx]._module.set_ob_mean_std(obstat.mean, obstat.std)
        ranker = CenteredRanker()
        # reporting
        if cfg.general.mlflow: mlflow_reporter.set_active_run(idx)
        reporter.start_gen()
        reporter.log({'idx': idx})
        # running es
        tr, gen_obstat = es.step(cfg, comm, population[idx], optims[idx], nt, env, eval_fn, rs, ranker, reporter)
        # sharing result and obstat
        tr = comm.scatter([tr] * comm.size)
        gen_obstat.mpi_inc(comm)
        obstat += gen_obstat

        if exploring(gen):  # Updating novelty
            behv = comm.scatter([mean_behv(population[idx], ns_fn, cfg.novelty.rollouts)] * comm.size)
            nov = comm.scatter([novelty(behv, archive, cfg.novelty.k)] * comm.size)
            archive = update_archive(comm, behv, archive)  # adding new behaviour and sharing archive
            policies_novelties[idx] = nov
            reporter.log({'nov': nov})
            # Saving best individuals for use later
            if tr.reward > policies_best_rewards[idx]:
                policies_best_rewards[idx] = tr.reward
                best_policies_params = np.copy(population[idx].flat_params)

        # Switching to reward function and starting from the best performing individuals found through novelty search
        if gen == cfg.sner.n_explorations:
            reporter.print(f'Reached generation {gen}. Updating all policy parameters and switching to reward function')
            eval_fn = r_fn
            for i in range(len(population)):
                population[i].set_nn_params(best_policies_params)

        dist = np.linalg.norm(np.array(tr.positions[-3:-1]))
        rew = tr.reward

        # Saving policy if it obtained a better reward or distance
        if (rew > best_rew or dist > best_dist) and comm.rank == 0:
            best_rew = max(rew, best_rew)
            best_dist = max(dist, best_dist)
            population[idx].save(f'saved/{cfg.general.name}', str(gen))
            reporter.print(f'saving policy with rew:{rew:0.2f} and dist:{dist:0.2f}')

        reporter.end_gen()

    mlflow.end_run()  # ending the outer mlfow run
