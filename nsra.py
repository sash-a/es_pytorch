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
from es.utils.ranking_functions import CenteredRanker, MultiObjectiveRanker, GBestRanker
from es.utils.reporters import LoggerReporter, ReporterSet, StdoutReporter, MLFlowReporter
from es.utils.training_result import NSRResult, NSResult, SavingResult
from es.utils.utils import generate_seed


def mean_behv(policy: Policy, r_fn: Callable[[torch.nn.Module, gym.Env], NSResult], e: gym.Env, rollouts: int):
    behvs = [r_fn(policy.pheno(np.zeros(len(policy))), e).behaviour for _ in range(rollouts)]
    return np.mean(behvs, axis=0)


if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD
    gym.logger.set_level(40)

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    # seeding
    cfg.general.seed = (generate_seed(comm) if cfg.general.seed is None else cfg.general.seed)
    torch.random.manual_seed(cfg.general.seed)
    rs = np.random.RandomState(cfg.general.seed + 10000 * comm.rank)

    mlflow_reporter = MLFlowReporter(comm, cfg_file, cfg)
    reporter = ReporterSet(
        LoggerReporter(comm, cfg, cfg.general.name),
        StdoutReporter(comm),
        mlflow_reporter
    )
    reporter.print(f'seed:{cfg.general.seed}')

    env: gym.Env = gym.make(cfg.env.name)
    # init population
    in_size, out_size = np.prod(env.observation_space.shape), np.prod(env.action_space.shape)
    population = []
    nns = []
    for _ in range(cfg.general.n_policies):
        nns.append(FullyConnected(int(in_size), int(out_size), 256, 2, torch.nn.Tanh(), env, cfg.policy))
        population.append(Policy(nns[-1], cfg.noise.std))
    # init optimizer and noise table
    optims: List[Optimizer] = [Adam(policy, cfg.policy.lr) for policy in population]
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(population[0]), reporter, cfg.general.seed)

    obstat: ObStat = ObStat(env.observation_space.shape, 1e-2)  # eps to prevent dividing by zero at the beginning

    archive: Optional[np.ndarray] = None
    policies_novelties = []

    best_rew = -np.inf
    best_dist = -np.inf


    def ns_fn(model: torch.nn.Module, e: gym.Env, max_steps: int = 2000, r: np.random.RandomState = None) -> NSRResult:
        """Reward function"""
        rews, behv, obs, steps = gym_runner.run_model(model, e, max_steps, r)
        # return RewardResult(rews, behv, obs, steps)
        return NSRResult(rews, behv, obs, steps, archive, cfg.novelty.k)


    # initializing the archive and policy weighting
    for policy in population:
        behv = None
        nov = None
        if comm.rank == 0:
            behv = mean_behv(policy, ns_fn, env, cfg.novelty.rollouts)
        archive = update_archive(comm, behv, archive)
        behv = archive[-1]
        nov = max(1e-2, novelty(behv, archive, cfg.novelty.k))
        policies_novelties.append(nov)

    # init global and local bests
    local_bests = [SavingResult(policy.flat_params + np.array([1e-4] * len(policy)),
                                ns_fn(policy.pheno(np.zeros(len(policy))), env).result, 1) for policy in population]
    global_best = SavingResult(population[0].flat_params + np.array([1e-4] * len(population[0])), local_bests[0].fit, 1)

    for gen in range(cfg.general.gens):  # main loop
        # picking the policy from the population
        # idx = random.choices(list(range(len(policies_novelties))), weights=policies_novelties, k=1)[0]
        # idx = comm.scatter([idx] * comm.size)
        idx = gen % cfg.general.n_policies  # round robin selection
        nns[idx].set_ob_mean_std(obstat.mean, obstat.std)
        ranker = GBestRanker(MultiObjectiveRanker(CenteredRanker(), -1))
        # reporting
        mlflow_reporter.set_active_run(idx)
        reporter.start_gen()
        reporter.log({'idx': idx})
        reporter.log({'w': population[idx].w})
        # running es
        tr, gen_obstat, gen_best = es.step(cfg, comm, population[idx], optims[idx], nt, env, ns_fn, global_best,
                                           local_bests[idx], rs, ranker, reporter)
        # sharing result and obstat
        tr = comm.scatter([tr] * comm.size)
        gen_obstat.mpi_inc(comm)
        obstat += gen_obstat
        # updating the weighting for choosing the next policy to be evaluated
        behv = comm.scatter([mean_behv(population[idx], ns_fn, env, cfg.novelty.rollouts)] * comm.size)
        nov = comm.scatter([novelty(behv, archive, cfg.novelty.k)] * comm.size)
        archive = update_archive(comm, behv, archive)  # adding new behaviour and sharing archive
        policies_novelties[idx] = nov

        dist = np.linalg.norm(np.array(tr.positions[-3:-1]))
        rew = tr.reward

        reporter.print(f'[out]gbest rank:{global_best.fit}')
        reporter.print(f'[out]lbest rank:{local_bests[idx].fit}')

        local_bests[idx] = max(local_bests[idx], gen_best, key=lambda x: x.fit[0])
        global_best = max(global_best, gen_best, key=lambda x: x.fit[0])

        # Saving policy if it obtained a better reward or distance
        if (rew > best_rew or dist > best_dist) and comm.rank == 0:
            best_rew = max(rew, best_rew)
            best_dist = max(dist, best_dist)
            population[idx].save(f'saved/{cfg.general.name}', str(gen))
            reporter.print(f'saving policy with rew:{rew:0.2f} and dist:{dist:0.2f}')

        reporter.end_gen()

    mlflow.end_run()  # ending the outer mlfow run
