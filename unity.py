from typing import Tuple, List, Optional

import gym
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


# channel = EngineConfigurationChannel()
# env: UnityEnvironment = UnityEnvironment("ml-agents-release_12/envs/soccer/headfull/soccer.x86_64", 0,
#                                          no_graphics=True, side_channels=[channel])
# env.reset()
# print('\n\n\n\nHERE')
#
# for k, v in env.behavior_specs.items():
#     k: BehaviorMapping
#     v: BehaviorMapping
#     print(f'k:{k} - v:{v}')
#
# print('\n\n\n\n')
#
# a: Tuple[DecisionSteps, TerminalSteps] = env.get_steps('Goalie?team=1')
# # a: Tuple[DecisionSteps, TerminalSteps] = env.get_steps('Striker?team=0')
# print(a[0].obs[0].shape)
# print(a[0].reward)
#
# env.set_actions('Goalie?team=1', np.zeros((8, 3)))
# env.set_actions('Striker?team=0', np.zeros((16, 3)))
# env.step()
#
#
# # channel.set_configuration_parameters(time_scale=50.0)
# # env = UnityToGymWrapper(unity_env)


class UnityGymWrapper(gym.Env):
    GymResult = Tuple[List[np.ndarray], List[np.ndarray], bool, dict]

    def __init__(self, name: Optional[str], rank: int, render=False, time_scale=50.):
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=time_scale)
        self._e: UnityEnvironment = UnityEnvironment(name, rank, no_graphics=not render, side_channels=[channel])
        self._e.reset()

        self.behaviour_names = list(self._e.behavior_specs.keys())
        print(self.behaviour_names)

    def step(self, actions: List[np.ndarray]) -> GymResult:
        for behaviour_name, action in zip(self.behaviour_names, actions):
            self._e.set_actions(behaviour_name, action)

        return self.collect_obs()

    def reset(self) -> GymResult:
        self._e.reset()
        return self.collect_obs()

    def collect_obs(self) -> GymResult:
        """
        :returns a list of observations. Each list item belongs to a different team. Within each item there may be is an
        ndarry of observations, where each dimension is the observation of a team member.
        """
        obs = []
        rews = []
        done = False

        for name in self.behaviour_names:
            step, term_step = self._e.get_steps(name)
            if term_step:
                done = True
                obs += term_step.obs
                rews += term_step.reward
            else:
                done = False
                obs += step.obs
                rews += [step.reward]

        return obs, rews, done, {}

    def render(self, mode='human'):
        raise Warning('Render cannot be called for unity env, it must be set in the constructor')


e = UnityGymWrapper(None, 0, render=True, time_scale=1.)
done = False
while not done:
    ob, _, done, _ = e.step([np.ones((1, 3)), np.ones((2, 3))])
    print(ob)
