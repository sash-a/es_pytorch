from typing import Tuple, List, Optional

import gym
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


class UnityGymWrapper(gym.Env):
    GymResult = Tuple[List[np.ndarray], List[np.ndarray], bool, dict]

    def __init__(self, name: Optional[str], rank: int, max_steps=2000, render=False, time_scale=50.):
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=time_scale)

        self.n = 0  # number of steps
        self.max_steps = max_steps

        self._e: UnityEnvironment = UnityEnvironment(name, rank, no_graphics=not render, side_channels=[channel])
        self._e.reset()

        self.behaviour_names = list(self._e.behavior_specs.keys())

    def step(self, actions: List[np.ndarray]) -> GymResult:
        for behaviour_name, action in zip(self.behaviour_names, actions):
            self._e.set_actions(behaviour_name, ActionTuple(discrete=action))

        self._e.step()
        self.n += 1

        return self.collect_obs()

    def reset(self) -> GymResult:
        self._e.reset()
        self.n = 0
        return self.collect_obs()

    def collect_obs(self) -> GymResult:
        """
        :returns a list of observations. Each list item belongs to a different team. Within each item there may be is an
        ndarry of observations, where each dimension is the observation of a team member.
        """
        obs = []
        rews = []
        done = self.n >= self.max_steps

        for name in self.behaviour_names:
            decision_step, term_step = self._e.get_steps(name)

            step = term_step if term_step else decision_step
            obs += step.obs
            rews += [step.reward]
            done = bool(term_step) or done

        return obs, rews, done, {'step': term_step if done else decision_step}

    def render(self, mode='human'):
        raise Warning('Render cannot be called for unity env, it must be set in the constructor')


if __name__ == '__main__':
    print('starting...')
    e = UnityGymWrapper(None, 0, max_steps=2000, render=True, time_scale=1.)
    done = False
    while not done:
        strikers = np.random.randint(-1, 2, (2, 3), dtype=np.int)
        goalie = np.random.randint(-1, 2, (1, 3), dtype=np.int)

        ob, rew, done, _ = e.step([strikers, goalie])
    print('done')
