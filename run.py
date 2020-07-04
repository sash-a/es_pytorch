import torch

from genome import Genome
from noisetable import NoiseTable

import numpy as np
import gym


def run():
    evals = 1

    nt = NoiseTable(table_size=1000000)
    layer_sizes = [(4, 64), (64, 64), (64, 64), (64, 2)]
    genome = Genome(layer_sizes, nt)

    env = gym.make('CartPole-v0')
    max_steps = 1000

    for i in range(evals):
        seed = np.random.randint(0, len(nt))
        model = genome.to_pheno(seed)

        obs = env.reset()

        with torch.no_grad():
            for _ in range(max_steps):
                obs = torch.squeeze(torch.from_numpy(obs).float())

                action = model(obs).argmax(0).item()
                obs, rew, done, _ = env.step(action)
                genome.fitness += rew

    env.close()


if __name__ == '__main__':
    run()
