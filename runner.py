import torch

from genome import Genome
from noisetable import NoiseTable

import gym


def run_genome(geno: Genome, noise_table: NoiseTable, render=False):
    env = gym.make('CartPole-v0')
    model, noise_idx = geno.pheno(noise_table)

    fitness = run_model(model, env, render)

    env.close()

    return fitness, noise_idx


def run_model(model: torch.nn.Module, env: gym.Env, render):
    max_steps = 1000
    fitness = 0

    obs = env.reset()

    with torch.no_grad():
        for _ in range(max_steps):
            obs = torch.from_numpy(obs).float()

            action = model(obs).argmax(0).item()
            obs, rew, done, _ = env.step(action)
            fitness += rew

            if render:
                env.render()

            if done:
                break

    return fitness
