# registering all envs that aren't included in gym
try:
    import pybullet_envs
except:
    pass

try:
    import pybulletgym
except:
    pass

try:
    import hrl_pybullet_envs
except:
    pass

try:
    import src.hbaselines.envs
except:
    pass
