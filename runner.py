import torch
import gym
import pybullet_envs

from genome import Genome
from noisetable import NoiseTable


def run_genome(geno: Genome, noise_table: NoiseTable, env_name: str, render: bool = False):
    env = gym.make(env_name, render=render)
    model, noise_idx = geno.pheno(noise_table)

    fitness = run_model(model, env, render)

    env.close()

    return fitness, noise_idx


def run_model(model: torch.nn.Module, env: gym.Env, render: bool):
    max_steps = 2000
    fitness = 0

    obs = env.reset()

    with torch.no_grad():
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
