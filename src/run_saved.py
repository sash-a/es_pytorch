import gym
import numpy as np

from es.policy import Policy
from utils.gym_runner import run_model

if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import pybullet_envs

    e = gym.make('HopperBulletEnv-v0', render=True).unwrapped
    r, d = run_model(Policy.load('../saved/dist/policy-1000').pheno(np.zeros(136451)), e, 2000, render=True)
    print(f'\n\nrewards {r}\ndist {d}\n\n')
    e.close()
