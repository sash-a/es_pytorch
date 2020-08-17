import gym
import numpy as np

from es.policy import Policy
from utils.gym_runner import run_model


def run_saved(policy_path: str, env, steps, ):
    p = Policy.load(policy_path)
    r, d = run_model(p.pheno(np.zeros(len(p))), env, steps, render=True)
    print(f'\n\nrewards {r}\ndist {d}\n\n')


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import pybullet_envs

    e = gym.make('Walker2DBulletEnv-v0', render=True).unwrapped
    run_saved('../saved/walker_rew/policy-2500', e, 2000)
    e.close()
