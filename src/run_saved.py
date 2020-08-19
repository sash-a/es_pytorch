import gym
import numpy as np

from es.policy import Policy
from utils.gym_runner import run_model


def run_saved(policy_path: str, env, steps, ):
    p = Policy.load(policy_path)
    r, d = run_model(p.pheno(np.zeros(len(p))), env, steps, render=True)
    print(f'\n\nrewards {sum(r)}\ndist {np.linalg.norm(np.array(d[:3]) - np.array(d[-3:]))}\n\n')


if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import pybullet_envs

    e = gym.make('HumanoidBulletEnv-v0', render=True).unwrapped
    run_saved('../saved/hum_dist/policy-9900', e, 2000)
    e.close()
