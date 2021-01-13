import os
from typing import List, Iterable

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
from es.utils.rankers import CenteredRanker
from es.utils.reporters import ReporterSet, LoggerReporter, StdoutReporter, MLFlowReporter
from es.utils.training_result import TrainingResult, MultiAgentTrainingResult, RewardResult
from es.utils.unity import UnityGymWrapper
from es.utils.utils import generate_seed


def inc_obstats(obstats: List[ObStat], results: Iterable[MultiAgentTrainingResult]):
    obs_sums_sqs_cnts = [r.ob_sum_sq_cnt for r in results]
    for i, ob in enumerate(obstats):
        for sum_sq_cnt in obs_sums_sqs_cnts:
            ob.inc(*sum_sq_cnt[i])

    return obstats


def custom_test_params(n: int, policies: List[Policy], fit_fn, obstats: List[ObStat]):
    results_pos, results_neg, all_inds = [], [], []
    results_pos: List[MultiAgentTrainingResult]
    results_neg: List[MultiAgentTrainingResult]

    for i in range(n):
        nns = []
        iter_inds = []
        for policy in policies:  # creating neural nets for all agents given the noise\
            idx, noise = nt.sample(rs)
            iter_inds.append(idx)
            nns += [policy.pheno(noise)]

        # for each noise ind sampled, both add and subtract the noise
        all_inds.append(iter_inds)
        results_pos.append(fit_fn(nns))
        results_neg.append(fit_fn(nns))

        obstats = inc_obstats(obstats, (results_pos[-1], results_neg[-1]))

    results = []
    steps = []
    n_objectives = 1  # todo

    for i, policy in policies:
        # collect positive/negative results + inds for each agent
        rp = [rp.trainingresults(RewardResult)[i] for rp in results_pos]
        rn = [rn.trainingresults(RewardResult)[i] for rn in results_neg]
        inds = [ind[i] for ind in all_inds]

        results.append(es._share_results(comm, [tr.result for tr in rp], [tr.result for tr in rn], inds))
        steps.append(comm.allreduce(sum([tr.steps for tr in rp + rn]), op=MPI.SUM))
        gen_obstats[i].mpi_inc(comm)

    return [(r[:, 0:n_objectives], r[:, n_objectives:2 * n_objectives], r[:, -1], s) for r, s in zip(results, steps)]


if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    full_name = f'{os.path.basename(cfg.env.name).split(".")[0]}-{cfg.general.name}'
    mlflow_reporter = MLFlowReporter(comm, cfg) if cfg.general.mlflow else None
    reporter = ReporterSet(
        LoggerReporter(comm, full_name),
        StdoutReporter(comm),
        mlflow_reporter
    )
    env: UnityGymWrapper = UnityGymWrapper(cfg.env.name, comm.rank, max_steps=50, render=False, time_scale=10.)

    # seeding; this must be done before creating the neural network so that params are deterministic across processes
    cfg.general.seed = (generate_seed(comm) if cfg.general.seed is None else cfg.general.seed)
    rs = utils.seed(comm, cfg.general.seed, env)

    # initializing obstat, policy, optimizer, noise and ranker
    obstats: List[ObStat] = [ObStat(env.observation_space[i].shape, 1e-2) for i in range(2)]
    neuralnets = [FullyConnected(int(np.prod(env.observation_space[i].shape)), int(np.prod(env.action_space[i].shape)),
                                 256, 2, torch.nn.Tanh(), env, cfg.policy) for i in range(2)]
    policies: List[Policy] = [Policy(nn, cfg.noise.std) for nn in neuralnets]
    optims: List[Optimizer] = [Adam(policy, cfg.policy.lr) for policy in policies]
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(policies[0]), None, cfg.general.seed)
    ranker = CenteredRanker()


    def r_fn(model: torch.nn.Module, use_ac_noise=True) -> TrainingResult:
        save_obs = rs.random() < cfg.policy.save_obs_chance
        rews, behv, obs, stps = gym_runner.run_model(model, env, cfg.env.max_steps, rs if use_ac_noise else None,
                                                     save_obs)
        return RewardResult(rews, behv, obs, stps)


    for _ in range(cfg.general.gens):
        gen_obstats = [ObStat(env.observation_space[i].shape, 0) for i in range(2)]
        results = custom_test_params(5, policies, r_fn, gen_obstats)
        for (pos_res, neg_res, inds, steps), policy, optim in zip(results, policies, optims):
            ranker.rank(pos_res, neg_res, inds)
            es.approx_grad(ranker, nt, policy.flat_params, optim, cfg.general.batch_size, cfg.policy.l2coeff)
            noiseless_result = r_fn(policy.pheno(np.zeros(len(policy))), False)
            reporter.log_gen(ranker.fits, noiseless_result, policy, steps)

    mlflow.end_run()  # in the case where mlflow is the reporter, just ending its run

print('done')
