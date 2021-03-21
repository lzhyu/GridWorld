"""
Fourrooms Game with water(extra walls)

class:
    + FourroomsWaterState
    + FourroomsWater
    + FourroomsWaterNorender

Properties:
Random waters garantee the rest space to be connected.
Resetting resets start position, goal position, coins and waters.
Each game has fixed number of coins and waters.

Possible extensions:
Magic blocks: transfer the agent to somewhere else with certain probability.
Each game has random number of coins and waters.
Discription of the model.
"""

import gym
import time
from gym import error, core, spaces
from gym.envs.registration import register
import random
import numpy as np
from fourrooms import *
from fourrooms_coin import *
from wrappers import ImageInputWarpper
from copy import deepcopy
import abc
import cv2
import time
from test_util import *


class FourroomsWaterState(FourroomsCoinState):
    def __init__(self, position_n: int, current_steps: int, goal_n: int, done: bool, num_pos: int, coin_dict: dict,
                 num_coins, water_list: list, num_waters, cum_reward: list):
        self.position_n = position_n
        self.current_steps = current_steps
        self.goal_n = goal_n
        self.done = done
        self.num_pos = num_pos
        self.coin_dict = coin_dict
        self.num_coins = num_coins
        self.water_list = water_list
        self.num_waters = num_waters
        self.cum_reward = cum_reward

    def __init__(self, base: FourroomsCoinState, water_list, num_waters):
        self.position_n = base.position_n
        self.current_steps = base.current_steps
        self.goal_n = base.goal_n
        self.done = base.done
        self.num_pos = base.num_pos
        self.coin_dict = base.coin_dict
        self.num_coins = base.num_coins
        self.water_list = water_list
        self.num_waters = num_waters
        self.cum_reward = base.cum_reward

    def watered_state(self):
        num_coins = len(self.coin_dict)
        value_list = [(v[0] if v[1] else 0) for v in self.coin_dict.values()]
        multiplier = np.dot(value_list, [2 ** i for i in range(num_coins)])
        return multiplier * self.num_pos + self.position_n

    def to_obs(self) -> np.array:
        return np.array(self.watered_state())


class FourroomsWater(FourroomsCoinNorender):
    def __init__(self, max_epilen=100, goal=None, num_coins=3, num_waters=3, seed=0):
        super(FourroomsCoin, self).__init__(max_epilen, goal, seed)
        self.num_waters = num_waters
        assert self.num_pos > (self.num_waters + 10), "too many waters."
        self.num_coins = num_coins
        assert (self.num_pos - self.num_waters) > (self.num_coins + 5), "too many coins."
        self.init_states = list(range(self.observation_space.n))
        self.init_states_ori = deepcopy(self.init_states)
        self.observation_space = spaces.Discrete((self.num_pos - self.num_waters) * (2 ** num_coins))
        self.occupancy_ori = deepcopy(self.occupancy)
        self.reset()

    def step_tmp(self, cell, action):
        nextcell = tuple(cell + self.directions[action])
        if not self.occupancy[nextcell]:
            return nextcell
        return cell

    def not_block(self, water):
        water_cell = self.tocell[water]
        self.occupancy[self.tocell[water]] = 1
        around = self.empty_around(water_cell)
        if len(around) == 0:
            return True
        spread = []
        remain = [around[0]]
        while len(around) != 0 and len(remain) != 0:
            currentcell = remain[0]
            remain.remove(currentcell)
            spread.append(currentcell)
            if currentcell in around:
                around.remove(currentcell)
            for action in range(self.action_space.n):
                nextcell = self.step_tmp(currentcell, action)
                if nextcell not in spread and nextcell not in remain:
                    remain.append(nextcell)
        if len(around) == 0:
            return True
        self.occupancy[self.tocell[water]] = 0
        return False

    def reset(self):
        self.init_states = deepcopy(self.init_states_ori)
        self.occupancy = deepcopy(self.occupancy_ori)
        water_list = np.array([], dtype=int)
        state_list = deepcopy(self.init_states)
        for _ in range(self.num_waters):
            while True:
                if len(state_list) == 0:
                    water = None
                    break
                water = np.random.choice(state_list)
                state_list.remove(water)
                if self.not_block(water):
                    break
            if water is not None:
                water_list = np.append(water_list, water)
                self.init_states.remove(water)
                state_list = deepcopy(self.init_states)
            else:
                raise NotImplementedError("Building waters error.")
        super().reset()
        self.state = FourroomsWaterState(self.state, water_list, self.num_waters)
        return self.state.to_obs()

    def render(self):
        pass


class FourroomsWaterNorender(FourroomsWater):
    def __init__(self, max_epilen=100, goal=None, num_coins=3, num_waters=3, seed=0):
        super().__init__(max_epilen, goal, num_coins, num_waters, seed)

    def render_water_blocks(self, blocks=[]):
        for water in self.state.water_list:
            x, y = self.tocell[water]
            blocks.append(self.make_block(x, y, (0, 1, 0)))
        for coin, count in self.state.coin_dict.items():
            x, y = self.tocell[coin]
            if count[1]:  # exist
                blocks.append(self.make_block(x, y, (0, 1, 1)))
        blocks.extend(self.make_basic_blocks())

        arr = self.render_with_blocks(self.origin_background, blocks)
        return arr

    def render(self, **kwargs):
        blocks = []
        return self.render_water_blocks(blocks)

def seeimg(img):
    cv2.startWindowThread()
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    env = ImageInputWarpper(FourroomsWaterNorender())
    check_render(env)
    check_run(env)
    print("Basic check finished.")
