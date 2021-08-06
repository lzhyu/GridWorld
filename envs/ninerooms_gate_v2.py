"""
Ninerooms Game with gates -- v2
All gates are in one color, whose type is given by the description.
"""

from .ninerooms_gate import *
from ..utils.env_utils.ninerooms_util import *

def random_train_model():
    Model = dict()
    for gate in gates_pos:
        Model[gate] = np.random.choice(['coin', 'water', 'wall'])
    return Model

def random_test_model():
    return random_train_model()

class NineroomsGateV2State(NineroomsGateState):
    def __init__(self, position_n, current_steps, goal_n, done, num_pos, gatedict, cum_reward: list, descr=None):
        # gatedict: gate_pos -> bool (exist or not)
        super(NineroomsGateV2State, self).__init__(position_n, current_steps, goal_n, done, num_pos, gatedict, cum_reward, descr)

    def to_obs(self):
        if self.gatedict is None or len(self.gatedict) == 0:
            return 0
        gate_list = [(k, v) for k, v in self.gatedict.items()]
        gate_list.sort(key=lambda ele: ele[0])
        mult_list = [1 if v else 0 for (_, v) in gate_list]
        multiplier = np.dot(mult_list, [2 ** i for i in range(len(mult_list))])
        return multiplier * self.num_pos + self.position_n

    @classmethod
    def frombase(cls, base:NineroomsGateState):
        return cls(base.position_n, base.current_steps, base.goal_n, base.done, base.num_pos, base.gatedict, base.cum_reward, base.descr)


class NineroomsGateV2(NineroomsGate):
    def __init__(self, Model=None, max_epilen=100, init_pos=None, goal=None, seed=None, mode='train'):
        super().__init__(Model=Model, max_epilen=max_epilen, init_pos=init_pos, goal=goal, seed=seed, mode=mode,
                         easy_dy=False, easy_env=False, fix_pos=False)
        self.state = NineroomsGateV2State.frombase(self.state)

    def to_descr(self):
        if self.Model is None or len(self.Model) == 0:
            return [0]*12
        gate_list = [(k, v) for k, v in self.Model.items()]
        gate_list.sort(key=lambda ele: ele[0])
        descr = [v for (k, v) in gate_list]
        return descr

    def reset(self):
        self.open = True

        if self.mode == 'train' and self.model_random:
            self.Model = random_train_model()
        elif self.model_random:
            self.Model = random_test_model()

        init_rooms = [0, 1, 2, 3, 5, 6, 7, 8]

        if self.init_random:
            init_room = np.random.choice(init_rooms)
            init_pos_n = np.random.choice(rooms_pos[init_room])
        else:
            init_pos_n = self.init_pos
            init_room = -1
            for i in range(len(rooms_pos)):
                if init_pos_n in rooms_pos[i]:
                    init_room = i
                    break
        if init_room in init_rooms:
            init_rooms.remove(init_room)

        if self.goal_random:
            goal_room = np.random.choice(init_rooms)
            goal_n = np.random.choice(rooms_pos[goal_room])
        else:
            goal_n = self.goal

        self.state = NineroomsGateV2State(init_pos_n, 0, goal_n, False, self.num_pos, init_gatedict, [], self.to_descr())
        return self.state.to_obs()

    def step(self, action):
        if self.state.done:
            raise Exception("Environment should be reseted")

        info = {}
        self.state.current_steps += 1

        next_pos = self.basic_step(self.state.position_n, self.directions[action])
        if next_pos in gates_pos:
            if self.Model[next_pos] == 'coin' and self.state.gatedict[next_pos]:
                reward = 1
                self.state.gatedict[next_pos] = False
            elif self.Model[next_pos] == 'water':
                reward = -1
            else:
                next_pos = self.state.position_n
                reward = -0.1
        elif next_pos == self.state.goal_n:
            reward = 10
        else:
            reward = -0.1

        if next_pos == self.state.goal_n or self.state.current_steps >= self.max_epilen:
            done = True
        else:
            done = False

        self.state.position_n = next_pos
        self.state.cum_reward.append(reward)
        self.state.done = done
        if done:
            info = {'episode': {'r': sum(self.state.cum_reward), 'l': self.state.current_steps}}

        return self.state.to_obs(), reward, done, info

    def render(self, mode=0):
        blocks = []
        for gate in gates_pos:
            if self.state.gatedict.get(gate, None):
                x, y = self.tocell[gate]
                blocks.append(self.make_block(x, y, (0, 1, 0)))
        blocks.extend(self.make_basic_blocks())
        arr = self.render_with_blocks(self.origin_background, blocks)
        return arr

if __name__ == '__main__':
    # basic test
    env_origin = ImageInputWarpper(NineroomsGateV2())
    check_render(env_origin)
    check_run(env_origin)