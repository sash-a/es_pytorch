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

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    env: gym.Env = gym.make(cfg.env.name)

    # seeding; this must be done before creating the neural network so that params are deterministic across processes
    rs, my_seed, global_seed = utils.seed(comm, cfg.general.seed, env)
    all_seeds = comm.alltoall([my_seed] * comm.size)  # simply for saving/viewing the seeds used on each proc
    print(f'seeds:{all_seeds}')

    # initializing obstat, policy, optimizer, noise and ranker
    nn = FeedForward(cfg.policy.layer_sizes, torch.nn.Tanh(), env, cfg.policy.ac_std, cfg.policy.ob_clip)
    policy: Policy = Policy(nn, cfg, Adam)
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(policy), None, cfg.general.seed)
    ranker = CenteredRanker()


    def r_fn(model: torch.nn.Module) -> TrainingResult:
        save_obs = (rs.random() if rs is not None else np.random.random()) < cfg.policy.save_obs_chance
        rews, behv, obs, steps = gym_runner.run_model(model, env, 10000, rs)
        return RewardResult(rews, behv, obs if save_obs else np.array([np.zeros(env.observation_space.shape)]), steps)


    assert cfg.general.policies_per_gen % comm.size == 0 and (cfg.general.policies_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.policies_per_gen / comm.size) / 2)
    for gen in range(cfg.general.gens):  # main loop
        if comm.rank == 0: print(f'Generation:{gen}')  # only print on one process

        # the block below is encapsulated in es.step(...), but this is more flexible. Example use can be seen in obj.py
        gen_obstat = ObStat(env.observation_space.shape, 0)  # for normalizing the observation space
        # obtaining the fitnesses from many perturbed policies
        pos_fits, neg_fits, inds, steps = es.test_params(comm, eps_per_proc, policy, nt, gen_obstat, r_fn, rs)
        policy.update_obstat(gen_obstat)
        ranker.rank(pos_fits, neg_fits, inds)  # ranking the fitnesses between -1 and 1
        # approximating the gradient and updating policy.flat_params (pseudo backprop)
        es.approx_grad(policy, ranker, nt, policy.flat_params, cfg.general.batch_size, cfg.policy.l2coeff)

        if comm.rank == 0: print(f'avg fitness:{np.mean(np.concatenate((pos_fits, neg_fits)))}\n\n')
        if gen % 10 and comm.rank == 0:  # save policy every 10 generations
            policy.save(f'saved/{cfg.general.name}', str(gen))
