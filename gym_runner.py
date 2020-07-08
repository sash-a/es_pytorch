import pickle

import torch
import gym

from policy import Policy


def run_model(model: torch.nn.Module, env: gym.Env, max_steps: int, episodes: int = 1, render: bool = False):
    fitness = 0

    with torch.no_grad():
        for _ in range(episodes):  # Does running each policy multiple times even help get more stable outputs?
            obs = env.reset()

            for _ in range(max_steps):
                obs = torch.from_numpy(obs).float()

                action = model(obs)
                obs, rew, done, _ = env.step(action)
                fitness += rew

                if render:
                    env.render()

                if done:
                    break

    return fitness


def load_model(file: str) -> torch.nn.Module:
    policy: Policy = pickle.load(open(file, 'rb'))
    return policy.set_module_params(policy._module, policy.flat_params)


if __name__ == '__main__':
    e = gym.make('HopperBulletEnv-v0')
    run_model(load_model('saved/genome-5000'), e, 10000, True)
    e.close()
