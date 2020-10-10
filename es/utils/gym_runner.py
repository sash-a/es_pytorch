from typing import List, Tuple

import gym
import numpy as np
import torch

BULLET_ENV_SUFFIX = 'BulletEnv'


def run_model(model: torch.nn.Module,
              env: gym.Env,
              max_steps: int,
              rs: np.random.RandomState = None,
              save_obs: bool = False,
              render: bool = False) -> Tuple[List[float], List[float], np.ndarray, int]:
    """
    Evaluates model on the provided env
    :returns: tuple of rewards earned and positions at each timestep position list is always of length `max_steps`
    """
    behv = []
    rews = []
    obs = []
    with torch.no_grad():
        ob = env.reset()
        for step in range(max_steps):
            ob = torch.from_numpy(ob).float()
            action = model(ob, rs=rs)
            ob, rew, done, _ = env.step(action)
            if save_obs:
                obs.append(ob)

            rews += [rew]
            # behv.extend(_get_pos(env.unwrapped))

            if render:
                env.render()

            if done:
                break

    if not save_obs:
        obs.append(np.zeros(ob.shape))
    behv = [0, 0, 0]
    behv += behv[-3:] * (max_steps - int(len(behv) / 3))  # extending the behaviour vector to have `max_steps` elements
    return rews, behv, np.array(obs), step


def _get_pos(env):
    if BULLET_ENV_SUFFIX in env.spec.id:  # bullet env
        return env.robot_body.get_pose()[:3]
    else:  # mujoco env
        model = env.model
        mass = np.reshape(model.body_mass, (-1, 1))
        xpos = env.data.xipos
        center = (np.sum(mass * xpos, 0) / np.sum(mass))
        return center[0], center[1], center[2]
