import gym
from gym.envs.registration import register

from .cliff_walking import *
from .frozen_lake import *

LAKE_STEP_REW = -0.1
LAKE_HOLE_REW = -10
LAKE_GOAL_REW = 10
CLIFF_STEP_REW = -1
CLIFF_FALL_REW = -100
CLIFF_GOAL_REW = 100


__all__ = ['RewardingFrozenLakeEnv', 'WindyCliffWalkingEnv']

register(
    id='RewardingFrozenLake4x4-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '4x4', 'step_rew': LAKE_STEP_REW, 'hole_rew': LAKE_HOLE_REW, 'goal_rew': LAKE_GOAL_REW},
)

register(
    id='RewardingFrozenLake8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'step_rew': LAKE_STEP_REW, 'hole_rew': LAKE_HOLE_REW, 'goal_rew': LAKE_GOAL_REW},
)

register(
    id='RewardingFrozenLake15x15-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '15x15', 'step_rew': LAKE_STEP_REW, 'hole_rew': LAKE_HOLE_REW, 'goal_rew': LAKE_GOAL_REW},
)

register(
    id='RewardingFrozenLake20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'step_rew': LAKE_STEP_REW, 'hole_rew': LAKE_HOLE_REW, 'goal_rew': LAKE_GOAL_REW},
)

register(
    id='RewardingFrozenLakeNoRewards4x4-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '4x4', 'rewarding': False, 'step_rew': LAKE_STEP_REW, 'hole_rew': LAKE_HOLE_REW, 'goal_rew': LAKE_GOAL_REW},
)

register(
    id='RewardingFrozenLakeNoRewards8x8-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '8x8', 'rewarding': False, 'step_rew': LAKE_STEP_REW, 'hole_rew': LAKE_HOLE_REW, 'goal_rew': LAKE_GOAL_REW},
)

register(
    id='RewardingFrozenLakeNoRewards15x15-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '15x15', 'rewarding': False, 'step_rew': LAKE_STEP_REW, 'hole_rew': LAKE_HOLE_REW, 'goal_rew': LAKE_GOAL_REW},
)

register(
    id='RewardingFrozenLakeNoRewards20x20-v0',
    entry_point='environments:RewardingFrozenLakeEnv',
    kwargs={'map_name': '20x20', 'rewarding': False, 'step_rew': LAKE_STEP_REW, 'hole_rew': LAKE_HOLE_REW, 'goal_rew': LAKE_GOAL_REW},
)

register(
    id='WindyCliffWalking-v0',
    entry_point='environments:WindyCliffWalkingEnv',
    kwargs={'step_rew': CLIFF_STEP_REW, 'fall_rew': CLIFF_FALL_REW, 'goal_rew': CLIFF_GOAL_REW},
)


#def get_rewarding_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

#    global LAKE_STEP_REW
#    global LAKE_HOLE_REW
#    global LAKE_GOAL_REW
#    LAKE_STEP_REW = step_rew
#    LAKE_HOLE_REW = hole_rew
#    LAKE_GOAL_REW = goal_rew

#    return gym.make('RewardingFrozenLake4x4-v0')


def get_rewarding_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

    global LAKE_STEP_REW
    global LAKE_HOLE_REW
    global LAKE_GOAL_REW
    LAKE_STEP_REW = step_rew
    LAKE_HOLE_REW = hole_rew
    LAKE_GOAL_REW = goal_rew

    return gym.make('RewardingFrozenLake8x8-v0')


#def get_large_rewarding_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

#    global LAKE_STEP_REW
#    global LAKE_HOLE_REW
#    global LAKE_GOAL_REW
#    LAKE_STEP_REW = step_rew
#    LAKE_HOLE_REW = hole_rew
#    LAKE_GOAL_REW = goal_rew

#    return gym.make('RewardingFrozenLake12x12-v0')


def get_large_rewarding_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

    global LAKE_STEP_REW
    global LAKE_HOLE_REW
    global LAKE_GOAL_REW
    LAKE_STEP_REW = step_rew
    LAKE_HOLE_REW = hole_rew
    LAKE_GOAL_REW = goal_rew

    return gym.make('RewardingFrozenLake15x15-v0')


#def get_large_rewarding_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

#    global LAKE_STEP_REW
#    global LAKE_HOLE_REW
#    global LAKE_GOAL_REW
#    LAKE_STEP_REW = step_rew
#    LAKE_HOLE_REW = hole_rew
#    LAKE_GOAL_REW = goal_rew

#    return gym.make('RewardingFrozenLake20x20-v0')


def get_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

    global LAKE_STEP_REW
    global LAKE_HOLE_REW
    global LAKE_GOAL_REW
    LAKE_STEP_REW = step_rew
    LAKE_HOLE_REW = hole_rew
    LAKE_GOAL_REW = goal_rew

    return gym.make('FrozenLake-v0')


#def get_rewarding_no_reward_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

#    global LAKE_STEP_REW
#    global LAKE_HOLE_REW
#    global LAKE_GOAL_REW
#    LAKE_STEP_REW = step_rew
#    LAKE_HOLE_REW = hole_rew
#    LAKE_GOAL_REW = goal_rew

#    return gym.make('RewardingFrozenLakeNoRewards4x4-v0')


def get_rewarding_no_reward_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

    global LAKE_STEP_REW
    global LAKE_HOLE_REW
    global LAKE_GOAL_REW
    LAKE_STEP_REW = step_rew
    LAKE_HOLE_REW = hole_rew
    LAKE_GOAL_REW = goal_rew

    return gym.make('RewardingFrozenLakeNoRewards8x8-v0')


#def get_large_rewarding_no_reward_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

#    global LAKE_STEP_REW
#    global LAKE_HOLE_REW
#    global LAKE_GOAL_REW
#    LAKE_STEP_REW = step_rew
#    LAKE_HOLE_REW = hole_rew
#    LAKE_GOAL_REW = goal_rew

#    return gym.make('RewardingFrozenLakeNoRewards12x12-v0')


def get_large_rewarding_no_reward_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

    global LAKE_STEP_REW
    global LAKE_HOLE_REW
    global LAKE_GOAL_REW
    LAKE_STEP_REW = step_rew
    LAKE_HOLE_REW = hole_rew
    LAKE_GOAL_REW = goal_rew

    return gym.make('RewardingFrozenLakeNoRewards15x15-v0')


#def get_large_rewarding_no_reward_frozen_lake_environment(step_rew = LAKE_STEP_REW, hole_rew = LAKE_HOLE_REW, goal_rew = LAKE_GOAL_REW):

#    global LAKE_STEP_REW
#    global LAKE_HOLE_REW
#    global LAKE_GOAL_REW
#    LAKE_STEP_REW = step_rew
#    LAKE_HOLE_REW = hole_rew
#    LAKE_GOAL_REW = goal_rew

#    return gym.make('RewardingFrozenLakeNoRewards20x20-v0')


def get_cliff_walking_environment(step_rew = CLIFF_STEP_REW, fall_rew = CLIFF_FALL_REW, goal_rew = CLIFF_GOAL_REW):

    global CLIFF_STEP_REW
    global CLIFF_FALL_REW
    global CLIFF_GOAL_REW
    CLIFF_STEP_REW = step_rew
    CLIFF_FALL_REW = fall_rew
    CLIFF_GOAL_REW = goal_rew

    return gym.make('CliffWalking-v0')


def get_windy_cliff_walking_environment(step_rew = CLIFF_STEP_REW, fall_rew = CLIFF_FALL_REW, goal_rew = CLIFF_GOAL_REW):

    global CLIFF_STEP_REW
    global CLIFF_FALL_REW
    global CLIFF_GOAL_REW
    CLIFF_STEP_REW = step_rew
    CLIFF_FALL_REW = fall_rew
    CLIFF_GOAL_REW = goal_rew

    return gym.make('WindyCliffWalking-v0')

