"""Init file for all environments."""
import gym

from src.hbaselines.envs.efficient_hrl.envs import AntMaze, AntPush, AntFall

__all__ = ["AntMaze", "AntPush", "AntFall"]

for env in __all__:
     gym.envs.register(
          id=f'{env}-v0',
          entry_point=f'src.hbaselines.envs.efficient_hrl.envs:{env}',
          max_episode_steps=2000,
     )
