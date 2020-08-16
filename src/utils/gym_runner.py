from typing import List, Tuple

import gym
import numpy as np
import torch


def run_model(model: torch.nn.Module,
              env: gym.Env,
              max_steps: int,
              rs: np.random.RandomState = None,
              episodes: int = 1,
              render: bool = False) -> Tuple[float, List[float]]:
    dists = []
    fitness = 0

    with torch.no_grad():
        for _ in range(episodes):  # Does running each policy multiple times even help get more stable outputs?
            obs = env.reset()

            for _ in range(max_steps):
                obs = torch.from_numpy(obs).float()

                action = model(obs, rs=rs)
                obs, rew, done, _ = env.step(action)
                fitness += rew

                if render:
                    env.render()

                if done:
                    break

            dists.append(_get_pos(env)[0])

    return fitness / episodes, dists


def model_reward(model: torch.nn.Module,
                 env: gym.Env,
                 max_steps: int,
                 rs: np.random.RandomState = None,
                 episodes: int = 1,
                 render: bool = False) -> float:
    return run_model(model, env, max_steps, rs, episodes, render)[0]


def model_dist(model: torch.nn.Module,
               env: gym.Env,
               max_steps: int,
               rs: np.random.RandomState = None,
               episodes: int = 1,
               render: bool = False) -> float:
    return run_model(model, env, max_steps, rs, 1, render)[1][0]


def _get_pos(env: gym.Env):
    return env.robot_body.get_pose()[:3]
