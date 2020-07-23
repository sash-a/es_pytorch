import gym
import numpy as np
import torch


def run_model(model: torch.nn.Module,
              env: gym.Env,
              max_steps: int,
              rs: np.random.RandomState = None,
              episodes: int = 1,
              render: bool = False):
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

    return fitness / episodes
