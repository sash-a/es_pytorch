import argparse

import gym
import numpy as np

from es.evo.policy import Policy
from es.utils.gym_runner import run_model


def run_saved(policy_path: str, env: gym.Env, steps: int):
    p = Policy.load(policy_path)
    r, d = run_model(p.pheno(np.zeros(len(p))), env, steps, render=True)
    print(f'\n\nrewards {np.sum(r)}\ndist {np.linalg.norm(np.array(d[-3:-1]))}\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('pickle_file')
    args = parser.parse_args()

    # noinspection PyUnresolvedReferences
    import pybullet_envs

    e = gym.make(args.env)
    run_saved(args.pickle_file, e, 1000)
    e.close()
