from typing import Tuple, List, Optional

import gym
import numpy as np
import torch
from gym import spaces
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from src.nn.nn import FeedForward


class UnityGymWrapper(gym.Env):
    GymResult = Tuple[np.ndarray, np.ndarray, bool, dict]

    def __init__(self, name: Optional[str], rank: int, max_steps=2000, render=False, time_scale=50.):
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=time_scale)

        self.n = 0  # number of steps
        self.max_steps = max_steps

        self._e: UnityEnvironment = UnityEnvironment(name, rank, no_graphics=not render, side_channels=[channel])
        self._e.reset()

        self.team_names = list(self._e.behavior_specs.keys())
        self.agent_per_team = {}
        for team, spec in self._e.behavior_specs.items():
            self.agent_per_team[team] = len(self._e.get_steps(team)[0].obs[0])

        # self._spec = next(iter(self._e.behavior_specs.values()))
        self._action_space = []
        # Set action spaces
        for team, spec in self._e.behavior_specs.items():
            agents_in_team = self.agent_per_team[team]
            if spec.action_spec.is_discrete():
                self.action_size = spec.action_spec.discrete_size
                branches = spec.action_spec.discrete_branches
                if spec.action_spec.discrete_size == 1:
                    self._action_space += [spaces.Discrete(branches[0])] * agents_in_team
                else:
                    self._action_space += [spaces.MultiDiscrete(branches)] * agents_in_team

            elif spec.action_spec.is_continuous():  # todo no shape
                self.action_size = spec.action_spec.continuous_size
                high = np.array([1] * spec.action_spec.continuous_size)
                self._action_space += [spaces.Box(-high, high, dtype=np.float32)] * agents_in_team
            else:
                raise Exception(
                    "The gym wrapper does not provide explicit support for both discrete and continuous actions.")
        self._action_space = spaces.Tuple(self._action_space)

        # Set observations space
        obs_spaces = []
        for team, spec in self._e.behavior_specs.items():
            agents_in_team = self.agent_per_team[team]
            high = np.array([np.inf] * self._get_vec_obs_size(spec))  # vector observation is last
            # sums observation_shapes because each sensor passes its obs separately and it's flatten into a single list
            obs_spaces += [spaces.Box(-high, high, shape=(np.sum(spec.observation_shapes),))] * agents_in_team
        self._observation_space = spaces.Tuple(obs_spaces)

    observation_space = property(lambda self: self._observation_space)
    action_space = property(lambda self: self._action_space)

    def _get_vec_obs_size(self, spec) -> int:
        result = 0
        for obs_shape in spec.observation_shapes:
            if len(obs_shape) == 1:
                result += obs_shape[0]
        return result

    def step(self, actions: List[np.ndarray]) -> GymResult:  # todo add support for ActionTuple(continuous, discrete)
        curr_action_idx = 0
        for team in self.team_names:
            # print(f'start idx={curr_action_idx}. End idx = {curr_action_idx + self.agent_per_team[team]}')
            # print(f'len action list:{len(actions[curr_action_idx:curr_action_idx + self.agent_per_team[team]])}')
            action = np.vstack(actions[curr_action_idx:curr_action_idx + self.agent_per_team[team]])
            # print(f'actions shape:{action.shape}')
            self._e.set_actions(team, ActionTuple(action))
            curr_action_idx += self.agent_per_team[team]

        self._e.step()
        self.n += 1

        return self.collect_obs()

    def reset(self) -> np.ndarray:
        self._e.reset()
        self.n = 0
        return self.collect_obs()[0]

    def collect_obs(self) -> GymResult:
        """
        :returns a list of observations. Each list item belongs to a different team. Within each item there may be is an
        ndarry of observations, where each dimension is the observation of a team member.
        """
        obs = []
        rews = []
        done = self.n >= self.max_steps

        for name in self.team_names:
            decision_step, term_step = self._e.get_steps(name)

            step = term_step if len(term_step) != 0 else decision_step
            done = len(term_step) != 0 or done

            # if len(term_step) != 0:
            #     step.obs = step.obs[0]  # I *think* this is only used for multiple agents in a single scene
            # print(step.obs)
            obs += [[np.hstack(step.obs).squeeze()]]  # stacking the data from each sensor into single list

            rews += [step.reward]

        return np.concatenate(obs), np.concatenate(rews), done, {'step': term_step if done else decision_step}

    def render(self, mode='human'):
        raise Warning('Render cannot be called for unity env, it must be set in the constructor')


if __name__ == '__main__':
    print('starting...')
    env_path = '/home/sasha/Documents/es/ml-agents-release_12/envs/ant_gather/ant_gather.x86_64'
    # env_path = '/home/sasha/Documents/es/ml-agents-release_12/envs/soccer_ones/headfull/soccer_ones.x86_64'
    n_agents = 1

    e = UnityGymWrapper(env_path, 0, max_steps=2000, render=True, time_scale=1.)
    obs = e.reset()
    done = False
    nns = [FeedForward([256, 256], torch.nn.Tanh(), e, 0.02)]
    # a = Policy.load('../../saved/soccer_ones-homerun/weights/17/policy-0').pheno()
    # b = Policy.load('../../saved/soccer_ones-homerun/weights/17/policy-1').pheno()
    #
    # policies = [a, b]

    while not done:
        with torch.no_grad():
            actions = [nn(torch.from_numpy(ob).float(), rs=None) for ob, nn in zip(obs, nns)]
        obs, rew, done, _ = e.step(actions)
    print('done')
