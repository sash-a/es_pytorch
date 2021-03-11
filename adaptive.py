import gym
import numpy as np
import torch
from mpi4py import MPI

import src.core.es as es
from src.core.noisetable import NoiseTable
from src.core.policy import Policy
from src.gym import gym_runner
from src.gym.training_result import TrainingResult, RewardResult
from src.nn.nn import FeedForward
from src.nn.obstat import ObStat
from src.nn.optimizers import Adam
from src.utils import utils
from src.utils.rankers import CenteredRanker
from src.utils.reporters import DefaultMpiReporterSet, LoggerReporter, StdoutReporter

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    full_name = f'{cfg.env.name}-{cfg.general.name}'

    reporter = DefaultMpiReporterSet(comm, full_name, LoggerReporter(comm, full_name), StdoutReporter(comm))

    env: gym.Env = gym.make(cfg.env.name)

    # seeding; this must be done before creating the neural network so that params are deterministic across processes
    rs, my_seed, global_seed = utils.seed(comm, cfg.general.seed, env)
    all_seeds = comm.alltoall([my_seed] * comm.size)  # simply for saving/viewing the seeds used on each proc
    reporter.print(f'seeds:{all_seeds}')

    # initializing obstat, policy, optimizer, noise and ranker
    nn = FeedForward(cfg.policy.layer_sizes, torch.nn.Tanh(), env, cfg.policy.ac_std, cfg.policy.ob_clip)
    policy: Policy = Policy(nn, cfg.noise.std, Adam(len(Policy.get_flat(nn)), cfg.policy.lr))
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(policy), None, cfg.general.seed)
    ranker = CenteredRanker()

    tau = 1 + 2 / np.sqrt(len(policy))


    def r_fn(model: torch.nn.Module) -> TrainingResult:
        save_obs = (rs.random() if rs is not None else np.random.random()) < cfg.policy.save_obs_chance
        rews, behv, obs, steps = gym_runner.run_model(model, env, 10000, rs)
        return RewardResult(rews, behv, obs if save_obs else np.array([np.zeros(env.observation_space.shape)]), steps)


    assert cfg.general.policies_per_gen % comm.size == 0 and (cfg.general.policies_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.policies_per_gen / comm.size) / 2)
    for gen in range(cfg.general.gens):  # main loop
        if comm.rank == 0: reporter.print(f'Generation:{gen}')  # only print on one process

        # the block below is encapsulated in es.step(...), but this is more flexible. Example use can be seen in obj.py
        gen_obstat = ObStat(env.observation_space.shape, 0)  # for normalizing the observation space

        # testing two different stds and using the one which performs better on average
        mod_frac = np.exp(1 / 100 * rs.uniform()) if comm.rank == 0 else None
        mod_frac = comm.scatter(mod_frac)

        std_hi = policy.std * mod_frac  # tau
        std_lo = policy.std / mod_frac  # tau
        policy.std = std_hi
        pos_fits_hi, neg_fits_hi, inds_hi, _ = es.test_params(comm, eps_per_proc, policy, nt, gen_obstat, r_fn, rs)
        policy.std = std_lo
        pos_fits_lo, neg_fits_lo, inds_lo, _ = es.test_params(comm, eps_per_proc, policy, nt, gen_obstat, r_fn, rs)

        pos_fits = np.concatenate([pos_fits_hi, pos_fits_lo])
        neg_fits = np.concatenate([neg_fits_hi, neg_fits_lo])
        inds = np.concatenate([inds_hi, inds_lo])

        avg_hi = np.mean(np.concatenate([pos_fits_hi, neg_fits_hi])[:, 0])
        avg_lo = np.mean(np.concatenate([pos_fits_lo, neg_fits_lo])[:, 0])
        policy.std = std_hi if avg_hi > avg_lo else std_lo

        policy.update_obstat(gen_obstat)
        ranker.rank(pos_fits, neg_fits, inds)  # ranking the fitnesses between -1 and 1
        # approximating the gradient and updating policy.flat_params (pseudo backprop)
        es.approx_grad(policy, ranker, nt, policy.flat_params, cfg.general.batch_size, cfg.policy.l2coeff)

        if comm.rank == 0:
            reporter.print(f'std:{policy.std}')
            reporter.print(f'max:{np.max(np.concatenate((pos_fits, neg_fits))[:, 0])}\n'
                  f'avg:{np.mean(np.concatenate((pos_fits, neg_fits))[:, 0])}\n'
                  f'min:{np.min(np.concatenate((pos_fits, neg_fits))[:, 0])}\n\n')

        if gen % 10 and comm.rank == 0:  # save policy every 10 generations
            policy.save(f'saved/{cfg.general.name}', str(gen))
