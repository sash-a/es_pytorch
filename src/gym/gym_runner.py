import time
from typing import List, Tuple, Callable

import gym
import numpy as np
import torch

from src.gym.unity import UnityGymWrapper

BULLET_ENV_SUFFIX = 'BulletEnv'


def pybullet_envs_pos(env):  # pybullet_envs
    return env.robot.body_real_xyz


def pybullet_gym_pos(env):  # pybullet-gym
    return env.robot.robot_body.pose().xyz()


def hbaselines_pos(env):  # hbaselines
    return env.wrapped_env.get_body_com("torso")[:3]


def mujoco_pos(env):  # mujoco default envs
    model = env.model
    mass = np.reshape(model.body_mass, (-1, 1))
    xpos = env.data.xipos
    center = (np.sum(mass * xpos, 0) / np.sum(mass))
    return center[0], center[1], center[2]


def run_model(model: torch.nn.Module,
              env: gym.Env,
              max_steps: int,
              rs: np.random.RandomState = None,
              render: bool = False,
              get_pos_fn: Callable[[gym.Env], Tuple[float, float, float]] = pybullet_gym_pos) -> \
        Tuple[List[float], List[float], np.ndarray, int]:
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
            ob, rew, done, _ = env.step(action.numpy())
            rews += [rew]
            obs.append(ob)
            behv.extend(get_pos_fn(env.unwrapped))

            if render:
                env.render('human')
                time.sleep(1 / 60)  # if rendering only step about 60 times per second

            if done:
                break

    behv += behv[-3:] * (max_steps - int(len(behv) / 3))  # extending the behaviour vector to have `max_steps` elements
    return rews, behv, np.array(obs), step


def multi_agent_gym_runner(policies: List[torch.nn.Module],
                           env: UnityGymWrapper,
                           max_steps: int,
                           rs: np.random.RandomState = None,
                           save_obs: bool = False,
                           render: bool = False):
    rews = []
    saved_obs = []
    behv = []

    with torch.no_grad():
        obs = env.reset()
        for step in range(max_steps):
            # ob = torch.from_numpy(ob).float()

            actions = [(policy(torch.from_numpy(ob).float(), to_int=True)) for policy, ob in zip(policies, obs)]
            # actions: List[np.ndarray] = []
            # for team in policies:
            #     team_actions = []
            #     for policy in team:
            #         team_actions.append(policy(ob, rs=rs))
            #
            #     actions.append(np.array(team_actions))

            obs, rew, done, _ = env.step(actions)
            if save_obs:
                saved_obs += [obs]

            rews += [rew]
            behv.extend([0, 0, 0])  # todo

            if render:
                env.render()

            if done:
                break

    if not save_obs:
        saved_obs += [np.zeros(obs.shape)]

    behv += behv[-3:] * (max_steps - int(len(behv) / 3))  # extending the behaviour vector to have `max_steps` elements
    return rews, behv, np.array(saved_obs), step
