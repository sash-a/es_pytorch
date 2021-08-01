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

    timeout = 200
    world_size = 10
    enclosed = True
    tolerance = 1
    steps = 1000
    max_target_dist = 4
    max_targets = 0
    switch_flag_on_collision = False
    use_sensor = False

    e = gym.make(args.env,
                 enclosed=enclosed,
                 timeout=timeout,
                 size=world_size,
                 tolerance=tolerance,
                 max_target_dist=max_target_dist,
                 max_targets=max_targets,
                 switch_flag_on_collision=switch_flag_on_collision,
                 use_sensor=use_sensor,
                 debug=False).unwrapped

    e.mpi_common_rand = np.random.RandomState()
    AntGatherBulletEnv.ant_env_rew_weight = 1
    AntGatherBulletEnv.path_rew_weight = 0
    AntGatherBulletEnv.dist_rew_weight = 0

    e.render('human')
    e.reset()
    if args.record:
        e.scene._p.startStateLogging(e.scene._p.STATE_LOGGING_VIDEO_MP4, '~/Documents/es/testvid.mp4')

    if args.pickle_file.endswith('.pt'):
        run_saved_pytorch(args.pickle_file, e, steps)
    else:
        run_saved_policy(args.pickle_file, e, steps)
    e.close()
