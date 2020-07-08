import pickle

import numpy as np
import torch
import gym
import pybullet_envs

from genome import Genome
from noisetable import NoiseTable


def run_genome(geno: Genome,
               noise_table: NoiseTable,
               env_name: str,
               max_steps: int,
               episodes: int = 1,
               render: bool = False):
    env = gym.make(env_name, render=render)
    model, noise_idx = geno.pheno(noise_table)

    fitness = run_model(model, env, max_steps, episodes, render)

    env.close()

    return fitness, noise_idx


def run_model(model: torch.nn.Module, env: gym.Env, max_steps: int, episodes: int = 1, render: bool = False):
    fitness = 0

    with torch.no_grad():
        for _ in range(episodes):  # Does running each policy multiple times even help get more stable outputs?
            obs = env.reset()

            for _ in range(max_steps):
                obs = torch.from_numpy(obs).float()

                action = model(obs)
                obs, rew, done, _ = env.step(action)
                fitness += rew

                if render:
                    env.render()

                if done:
                    break

    return fitness


def load_model(file: str) -> torch.nn.Module:
    genome: Genome = pickle.load(open(file, 'rb'))
    return Genome.make_pheno(genome, np.zeros(10000000))


if __name__ == '__main__':
    e = gym.make('HopperBulletEnv-v0')
    run_model(load_model('saved/genome-5000'), e, 10000, True)
    e.close()
