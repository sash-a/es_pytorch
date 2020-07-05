import pickle

import gym
import pybullet_envs
import numpy as np
import torch

from genome import Genome
from runner import run_model


def load_model(file: str) -> torch.nn.Module:
    genome: Genome = pickle.load(open(file, 'rb'))
    return Genome.make_pheno(genome, np.zeros(10000000))


if __name__ == '__main__':
    env = gym.make('HopperBulletEnv-v0')
    run_model(load_model('saved/genome-5000'), env, True)
    env.close()
