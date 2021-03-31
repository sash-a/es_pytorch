import time
from typing import Tuple, List

import gym
import numpy as np
# noinspection PyUnresolvedReferences
import pybullet_envs
# noinspection PyUnresolvedReferences
import pybulletgym
import torch
from mpi4py import MPI
from torch import Tensor, clamp, cat, nn

import src.core.es as es
from src.core.noisetable import NoiseTable
from src.core.policy import Policy
from src.gym.training_result import TrainingResult, RewardResult
from src.nn.nn import BaseNet
from src.nn.optimizers import Adam
from src.utils import utils
from src.utils.rankers import CenteredRanker
from src.utils.reporters import DefaultMpiReporterSet, StdoutReporter, LoggerReporter


class PrimFF(BaseNet):
    def __init__(self, layer_sizes: List[int], activation: nn.Module, obs_shape, ac_std: float, ob_clip=5):
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [nn.Linear(in_size, out_size), activation]

        super().__init__(layers, obs_shape, ob_clip)

        self._action_std = ac_std

    GOAL = 'goal'

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        rs = kwargs['rs']
        goal = kwargs[PrimFF.GOAL]

        inp = clamp((inp - self._obmean) / self._obstd, min=-self.ob_clip, max=self.ob_clip)
        a = self.model(cat((goal, inp)).float())  # adding goal in after clip and vbn
        if self._action_std != 0 and rs is not None:
            a += rs.randn(*a.shape) * self._action_std

        return a


def run_model(model: PrimFF,
              env: gym.Env,
              max_steps: int,
              rs: np.random.RandomState = None,
              render: bool = False) -> \
        Tuple[List[float], List[float], np.ndarray, int]:
    """
    Evaluates model on the provided env
    :returns: tuple of rewards earned and positions at each timestep position list is always of length `max_steps`
    """
    behv = []
    rews = []
    obs = []

    if rs is not None:
        goal = torch.tensor((rs.uniform(-1, 1), rs.uniform(-1, 1)))  # normalized goal
        goal_pos = goal.numpy() * 7
    else:
        rs = np.random.RandomState()
        goal = torch.tensor((rs.uniform(-1, 1), rs.uniform(-1, 1)))  # normalized goal
        goal_pos = goal.numpy() * 7
        rs = None

    with torch.no_grad():
        ob = env.reset()
        for step in range(max_steps):
            ob = torch.from_numpy(ob).float()

            action = model(ob, rs=rs, goal=goal)
            ob, rew, done, _ = env.step(action.numpy())

            pos = env.unwrapped.parts['torso'].get_position()

            obs.append(ob)
            behv.extend(pos)

            if render:
                env.render('human')
                time.sleep(1 / 60)  # if rendering only step about 60 times per second

            if done:
                break

        rews += [-((pos[0] - goal_pos[0]) ** 2 + (pos[1] - goal_pos[1]) ** 2)]

    behv += behv[-3:] * (max_steps - int(len(behv) / 3))  # extending the behaviour vector to have `max_steps` elements
    return rews, behv, np.array(obs), step


if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD

    cfg_file = utils.parse_args()
    cfg = utils.load_config(cfg_file)

    run_name = f'{cfg.env.name}-{cfg.general.name}'
    reporter = DefaultMpiReporterSet(comm, run_name, StdoutReporter(comm), LoggerReporter(comm, run_name))

    env: gym.Env = gym.make(cfg.env.name)

    # seeding; this must be done before creating the neural network so that params are deterministic across processes
    rs, my_seed, global_seed = utils.seed(comm, cfg.general.seed, env)
    all_seeds = comm.alltoall([my_seed] * comm.size)  # simply for saving/viewing the seeds used on each proc
    print(f'seeds:{all_seeds}')

    # initializing obstat, policy, optimizer, noise and ranker
    in_size = int(np.prod(env.observation_space.shape))
    out_size = int(np.prod(env.action_space.shape))
    nn = PrimFF([in_size + 2] + cfg.policy.layer_sizes + [out_size], torch.nn.Tanh(), in_size, cfg.policy.ac_std,
                cfg.policy.ob_clip)
    policy: Policy = Policy(nn, cfg.noise.std, Adam(len(Policy.get_flat(nn)), cfg.policy.lr))
    nt: NoiseTable = NoiseTable.create_shared(comm, cfg.noise.tbl_size, len(policy), None, cfg.general.seed)
    ranker = CenteredRanker()


    def r_fn(model: PrimFF, use_ac_noise=True) -> TrainingResult:
        save_obs = (rs.random() if rs is not None else np.random.random()) < cfg.policy.save_obs_chance
        rews, behv, obs, steps = run_model(model, env, 1000, rs if use_ac_noise else None)
        return RewardResult(rews, behv, obs if save_obs else np.array([np.zeros(env.observation_space.shape)]), steps)


    assert cfg.general.policies_per_gen % comm.size == 0 and (cfg.general.policies_per_gen / comm.size) % 2 == 0
    eps_per_proc = int((cfg.general.policies_per_gen / comm.size) / 2)
    for gen in range(cfg.general.gens):  # main loop
        reporter.start_gen()
        tr, gen_obstat = es.step(cfg, comm, policy, nt, env, r_fn, rs, ranker, reporter)
        policy.update_obstat(gen_obstat)
        reporter.end_gen()

        if gen % 10 and comm.rank == 0:  # save policy every 10 generations
            policy.save(f'saved/{cfg.general.name}', str(gen))
