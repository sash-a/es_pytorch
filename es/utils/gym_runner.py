from typing import List, Tuple

import gym
import numpy as np
import torch

behv_period = 10


def run_model(model: torch.nn.Module,
              env: gym.Env,
              max_steps: int,
              rs: np.random.RandomState = None,
              render: bool = False) -> Tuple[List[float], List[float]]:
    """
    Evaluates model on the provided env
    :returns: tuple of cumulative rewards and distances traveled
    """
    behv = []
    rews = []

    with torch.no_grad():
        obs = env.reset()
        for step in range(max_steps):
            obs = torch.from_numpy(obs).float()

            action = model(obs, rs=rs)
            obs, rew, done, _ = env.step(action)
            rews += [rew]
            if step % behv_period == 0:
                behv.extend(_get_pos(env))

            if render:
                env.render()

            if done:
                break

    behv += behv[-3:] * int((max_steps / behv_period) - len(behv) / 3)

    return rews, behv


def _get_pos(env: gym.Env):
    return env.robot_body.get_pose()[:3]
