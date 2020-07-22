import gym

from utils.gym_runner import run_model, load_model

if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    import pybullet_envs
    e = gym.make('HopperBulletEnv-v0', render=True).unwrapped
    run_model(load_model('../saved/saved/policy-4096'), e, 10000, render=True)
    e.close()
