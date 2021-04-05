import argparse

import gym
import numpy as np
import torch

import rand_pos as rp
# noinspection PyUnresolvedReferences
from rand_pos import PrimFF
from src.core.policy import Policy
from src.gym.gym_runner import run_model


def run_saved_policy(policy_path: str, env: gym.Env, steps: int, rand_pos):
    run_saved(Policy.load(policy_path).pheno(), env, steps, rand_pos)


def run_saved_pytorch(policy_path: str, env: gym.Env, steps: int, rand_pos):
    run_saved(torch.load(policy_path), env, steps, rand_pos)


def run_saved(model: torch.nn.Module, env, steps, rand_pos):
    while True:
        if rand_pos:
            goal = torch.tensor(rp.gen_goal(np.random.RandomState()))
            print(f'goal:{goal}')
            r, d, _, s = rp.run_model(model, env, steps, render=True, goal_normed=goal)
        else:
            r, d, _, s = run_model(model, env, steps, render=True)

        print(f'\n\nrewards {np.sum(r)}\ndist {np.linalg.norm(np.array(d[-3:-1]))}\n\n')


if __name__ == '__main__':
    gym.logger.set_level(40)

    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('pickle_file')
    parser.add_argument('--rand_pos', dest='rand_pos', action='store_true')
    parser.add_argument('--no-rand_pos', dest='rand_pos', action='store_false')
    parser.set_defaults(rand_pos=True)

    args = parser.parse_args()

    # # noinspection PyUnresolvedReferences
    # import pybullet_envs

    e = gym.make(args.env)
    e.render('human')
    steps = 500

    if args.pickle_file.endswith('.pt'):
        run_saved_pytorch(args.pickle_file, e, steps, args.rand_pos)
    else:
        run_saved_policy(args.pickle_file, e, steps, args.rand_pos)
    e.close()
