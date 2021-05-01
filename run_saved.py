import argparse

import gym
import numpy as np
import torch
# noinspection PyUnresolvedReferences
from hrl_pybullet_envs import AntGatherBulletEnv

import flagrun
# noinspection PyUnresolvedReferences
from flagrun import PrimFF
from src.core.policy import Policy


def run_saved_policy(policy_path: str, env: gym.Env, steps: int):
    run_saved(Policy.load(policy_path).pheno(), env, steps)


def run_saved_pytorch(policy_path: str, env: gym.Env, steps: int):
    run_saved(torch.load(policy_path), env, steps)


def run_saved(model: torch.nn.Module, env, steps):
    while True:
        r, d, _, s = flagrun.run_model(model, env, steps, render=True)
        print(f'\n\nrewards {np.sum(r)}\ndist {np.linalg.norm(np.array(d[-3:-1]))}\n\n')


if __name__ == '__main__':
    gym.logger.set_level(40)

    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('pickle_file')
    parser.add_argument('--record', dest='record', action='store_true')
    parser.set_defaults(record=False)

    args = parser.parse_args()

    # # noinspection PyUnresolvedReferences
    # import pybullet_envs

    e = gym.make(args.env, enclosed=True, timeout=-1).unwrapped
    e.mpi_common_rand = np.random.RandomState()

    e.render('human')
    e.reset()
    steps = 1000
    if args.record:
        e.scene._p.startStateLogging(e.scene._p.STATE_LOGGING_VIDEO_MP4, '~/Documents/es/testvid.mp4')

    if args.pickle_file.endswith('.pt'):
        run_saved_pytorch(args.pickle_file, e, steps)
    else:
        run_saved_policy(args.pickle_file, e, steps)
    e.close()
