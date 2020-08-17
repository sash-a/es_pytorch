from typing import List, Tuple

import gym
import numpy as np
import torch


def run_model(model: torch.nn.Module,
              env: gym.Env,
              max_steps: int,
              rs: np.random.RandomState = None,
              episodes: int = 1,
              render: bool = False) -> Tuple[List[float], List[float]]:
    """
    Evaluates model on the provided env
    :returns: tuple of cumulative rewards and distances traveled
    """
    dists = []
    fitness = []

    with torch.no_grad():
        for _ in range(episodes):  # Does running each policy multiple times even help get more stable outputs?
            obs = env.reset()
            ep_fitness = 0
            for _ in range(max_steps):
                obs = torch.from_numpy(obs).float()

                action = model(obs, rs=rs)
                obs, rew, done, _ = env.step(action)
                ep_fitness += rew

                if render:
                    env.render()

                if done:
                    break

            dists.append(_get_pos(env)[0])
            fitness.append(ep_fitness)

    return fitness, dists


def _get_pos(env: gym.Env):
    return env.robot_body.get_pose()[:3]
