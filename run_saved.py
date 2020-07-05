import pickle

import gym
import numpy as np

from genome import Genome
from runner import run_model


def load(file: str):
    genome: Genome = pickle.load(open(file, 'rb'))
    return Genome.make_pheno(genome, np.zeros(10000000))


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    run_model(load('genome-253000'), env, True)
    env.close()
