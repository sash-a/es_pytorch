import gym
import numpy as np

from es.evo.policy import Policy
from es.utils.gym_runner import run_model


def run_saved(policy_path: str, env: gym.Env, steps: int):
    p = Policy.load(policy_path)
    r, d = run_model(p.pheno(np.zeros(len(p))), env, steps, render=True)
    print(f'\n\nrewards {sum(r)}\ndist {np.linalg.norm(np.array(d[-3:-1]))}\n\n')


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import pybullet_envs

    e = gym.make('HopperBulletEnv-v0', render=True).unwrapped
    run_saved('saved/hopper_dist_new2/policy-582', e, 2000)
    e.close()
