import gym
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
from src.utils.rankers import CenteredRanker
from src.utils.utils import generate_seed

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    env: gym.Env = gym.make(cfg.env.name)

    # seeding; this must be done before creating the neural network so that params are deterministic across processes
    cfg.general.seed = (generate_seed(comm) if cfg.general.seed is None else cfg.general.seed)
    rs = utils.seed(comm, cfg.general.seed, env)

    # initializing obstat, policy, optimizer, noise and ranker
    obstat: ObStat = ObStat(env.observation_space.shape, 1e-2)  # eps to prevent dividing by zero at the beginning
    nn = FullyConnected(int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape)),
                        256, 2, torch.nn.Tanh(), env, cfg.policy)
    policy: Policy = Policy(nn, cfg.noise.std)
    optim: Optimizer = Adam(policy, cfg.policy.lr)
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(policy), None, cfg.general.seed)
    ranker = CenteredRanker()


    def r_fn(model: torch.nn.Module) -> TrainingResult:
        save_obs = (rs.random() if rs is not None else np.random.random()) < cfg.policy.save_obs_chance
        rews, behv, obs, steps = gym_runner.run_model(model, env, 10000, rs, save_obs)
        return RewardResult(rews, behv, obs, steps)


    assert cfg.general.policies_per_gen % comm.size == 0 and (cfg.general.policies_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.policies_per_gen / comm.size) / 2)
    for gen in range(cfg.general.gens):  # main loop
        if comm.rank == 0: print(f'Generation:{gen}')  # only print on one process
        nn.set_ob_mean_std(obstat.mean, obstat.std)  # for normalizing the observation space

        # the block below is encapsulated in es.step(...), but this is more flexible. Example use can be seen in obj.py
        gen_obstat = ObStat(env.observation_space.shape, 0)  # for normalizing the observation space
        # obtaining the fitnesses from many perturbed policies
        pos_fits, neg_fits, inds, steps = es.test_params(comm, eps_per_proc, policy, nt, gen_obstat, r_fn, rs)
        obstat += gen_obstat  # adding the new observations to the global obstat
        ranker.rank(pos_fits, neg_fits, inds)  # ranking the fitnesses between -1 and 1
        # approximating the gradient and updating policy.flat_params (pseudo backprop)
        es.approx_grad(ranker, nt, policy.flat_params, optim, cfg.general.batch_size, cfg.policy.l2coeff)

        if comm.rank == 0: print(f'avg fitness:{np.mean(np.concatenate((pos_fits, neg_fits)))}\n\n')
        if gen % 10 and comm.rank == 0:  # save policy every 10 generations
            policy.save(f'saved/{cfg.general.name}', str(gen))
