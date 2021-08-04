"""
Fourrooms with gates
One gate between two adjacent rooms
    Gate type: coin/water/wall
    Four gate with same color
    gates_pos = [25, 51, 62, 88]
Transition dynamic is basic, i.e. no further step and direction turning.
Model description is a vector with dim=4 in field ('coin', 'water', 'wall')

Init pos and goal are at diagonal rooms.
Wall number is at most 1, so the agent can reach the goal.
"""
from abc import ABC
from fourrooms import *
from wrappers import ImageInputWarpper
from test_util import *
import numpy as np
from fourrooms_util import *


class FourroomsGateState(FourroomsBaseState):
    def __init__(self, position_n, current_steps, goal_n, done, num_pos, cum_reward, description, gate_list):
        # description is a 4-dim list in field ('coin', 'water', 'wall')
        # gate_list is a 4-dim list in field (0, 1), 0 means a coin gate is eaten
        super(FourroomsGateState, self).__init__(position_n, current_steps, goal_n, done, num_pos)
        self.cum_reward = cum_reward
        self.descr = description
        self.gate_list = gate_list

    @classmethod
    def frombase(cls, base: FourroomsBaseState, cum_reward, description, gate_list):
        return cls(base.position_n, base.current_steps, base.goal_n, base.done, base.num_pos, cum_reward,
                   description, gate_list)

    @property
    def to_obs(self):
        if len(self.gate_list) == 0:  # initial 0
            return 0
        descr_value = [0 if gate == 'coin' else 1 if gate == 'water' else 2 for gate in self.descr]
        multiplier_descr = np.dot(descr_value, [3 ** i for i in range(4)])
        multiplier_gate = np.dot(self.gate_list, [2 ** i for i in range(4)])
        multiplier = multiplier_descr * 16 + multiplier_gate
        return np.array(multiplier * self.num_pos + self.position_n)

    def to_tuple(self):
        return self.position_n, self.current_steps, self.goal_n, self.done, tuple(self.gate_list)

class FourroomsGate(FourroomsNorender, ABC):
    def __init__(self, Model=None, max_epilen=100, goal=None, seed=None, mode='train', gate_list=None):
        super(FourroomsGate, self).__init__(max_epilen, goal, seed)
        # Model: dict pos -> gate type
        self.goal = goal
        self.Model = Model
        self.mode = mode  # 'train' or 'test'
        self.gate_list = gate_list or [1, 1, 1, 1]
        self.state = FourroomsGateState(0, 0, 0, False, 0, [], [], [])
        self.open = False
        self.random_model = False if Model is not None else True

    def reset(self):
        super().reset()
        # random chooce initial position and goal, in diagonal rooms
        init_room = np.random.choice([0, 1, 2, 3])
        init_pos = np.random.choice(rooms_pos[init_room])
        goal_pos = np.random.choice(rooms_pos[3 - init_room])
        self.state.position_n = init_pos
        self.state.goal_n = goal_pos
        self.goal = goal_pos

        self.gate_list = [1, 1, 1, 1]
        if self.random_model:
            if self.mode == 'train':
                self.Model = np.random.choice(train_models)
            else:
                self.Model = np.random.choice(test_models)
        self.state = FourroomsGateState.frombase(self.state, [], self.todecr(), self.gate_list)
        return self.state.to_obs

    def todecr(self):
        descr = [self.Model[pos] for pos in gates_pos]
        return descr

    def step(self, action):
        if self.state.done:
            raise Exception("Environment should be reseted")
        currentcell = self.tocell[self.state.position_n]
        try:
            nextcell = tuple(currentcell + self.directions[action])
        except TypeError:
            nextcell = tuple(currentcell + self.directions[action[0]])

        if not self.occupancy[nextcell] and self.Model.get(self.tostate[nextcell], None) != 'wall':
            currentcell = nextcell
        position_n = self.tostate[currentcell]

        if position_n == self.state.goal_n:
            reward = 10
        elif position_n in gates_pos and self.gate_list[gates_pos.index(position_n)] == 1:
            if self.Model[position_n] == 'water':
                reward = -1
            else:  # coin
                reward = 10
                self.gate_list[gates_pos.index(position_n)] = 0
            self.state.gate_list = self.gate_list
        else:
            reward = -0.1

        self.state.current_steps += 1
        self.state.done = (position_n == self.state.goal_n) or (self.state.current_steps >= self.max_epilen)
        self.state.position_n = position_n
        self.state.cum_reward.append(reward)
        info = {}

        if self.state.done:
            info = {'episode': {'r': np.sum(self.state.cum_reward), 'l': self.state.current_steps}}

        return self.state.to_obs, reward, self.state.done, info

    def render(self, mode=0):
        pass

class FourroomsGateNorender(FourroomsGate):
    def __init__(self, Model=None, max_epilen=100, goal=None, seed=None, mode='train', gate_list=None):
        super().__init__(Model, max_epilen, goal, seed, mode, gate_list)

    def render(self, mode=0):
        blocks = []
        for i in range(4):
            if self.gate_list[i] == 0:
                continue
            x, y = self.tocell[gates_pos[i]]
            blocks.append(self.make_block(x, y, (0, 1, 0)))
        blocks.extend(self.make_basic_blocks())
        return self.render_with_blocks(self.origin_background, blocks)
