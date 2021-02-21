import argparse

import gym
import numpy as np
import torch

from src.core.policy import Policy
from src.gym.gym_runner import run_model, BULLET_ENV_SUFFIX


def run_saved_policy(policy_path: str, env: gym.Env, steps: int):
    run_saved(Policy.load(policy_path).pheno(), env, steps)


def run_saved_pytorch(policy_path: str, env: gym.Env, steps: int):
    run_saved(torch.load(policy_path), env, steps)


def run_saved(model: torch.nn.Module, env, steps):
    while True:
        r, d, _, s = run_model(model, env, steps, render=True)
        print(f'\n\nrewards {np.sum(r)}\ndist {np.linalg.norm(np.array(d[-3:-1]))}\n\n')


if __name__ == '__main__':
    gym.logger.set_level(40)

    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('pickle_file')
    args = parser.parse_args()

    # noinspection PyUnresolvedReferences
    import pybullet_envs

    if BULLET_ENV_SUFFIX in args.env:
        e = gym.make(args.env, render=True)
        e.render('human')
    else:
        e = gym.make(args.env)

    if args.pickle_file.endswith('.pt'):
        run_saved_pytorch(args.pickle_file, e, 2000)
    else:
        run_saved_policy(args.pickle_file, e, 2000)
    e.close()
