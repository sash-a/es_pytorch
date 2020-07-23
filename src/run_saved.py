import gym
import numpy as np

from es.policy import Policy
from utils.gym_runner import run_model

if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import pybullet_envs

    e = gym.make('HopperBulletEnv-v0', render=True).unwrapped
    run_model(Policy.load('../saved/saved/policy-4096').pheno(np.zeros(1000000)), e, 10000, render=True)
    e.close()
