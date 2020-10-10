import argparse

import gym
import numpy as np
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from es.evo.policy import Policy
from es.utils.gym_runner import run_model


def run_saved(policy_path: str, env: gym.Env, steps: int):
    p = Policy.load(policy_path)
    while True:
        r, d, _, _ = run_model(p.pheno(np.zeros(len(p))), env, steps, render=True)
        print(f'\n\nrewards {np.sum(r)}\ndist {np.linalg.norm(np.array(d[-3:-1]))}\n\n')


if __name__ == '__main__':
    gym.logger.set_level(40)

    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('pickle_file')
    args = parser.parse_args()

    # noinspection PyUnresolvedReferences
    import pybullet_envs

    # if BULLET_ENV_SUFFIX in args.env:
    #     e = gym.make(args.env, render=True)
    # else:
    #     e = gym.make(args.env)
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(args.env, 0, no_graphics=False, side_channels=[channel])
    # channel.set_configuration_parameters(time_scale=10.0)
    e = UnityToGymWrapper(unity_env)

    run_saved(args.pickle_file, e, 2000)
    e.close()
