"""
Basic Ninerooms Game
3 * 3 rooms.
A corridor around the nine rooms, connected to the outer 8 rooms.
"""


from fourrooms import *
import numpy as np
import cv2

class NineroomsBaseState(FourroomsBaseState):
    def __init__(self, position_n, current_steps, goal_n, done, num_pos):
        super().__init__(position_n, current_steps, goal_n, done, num_pos)

    def to_obs(self):
        return np.array(self.position_n)

    def to_tuple(self):
        return self.position_n, self.current_steps, self.goal_n, self.done

class NineroomsBase(FourroomsNorender):
    def __init__(self, max_epilen=100, init_pos=None, goal=None, seed=0):
        super().__init__(max_epilen=max_epilen, goal=goal, seed=seed)
        self.init_pos = init_pos

    def init_layout(self):
        self.layout = """\
11111111111111111111
1                  1
1 111 1111 111 111 1
1 1    1    1    1 1
1 1         1      1
1      1         1 1
1 1    1    1    1 1
1 1 111111 11111 1 1
1 1    1    1    1 1
1      1    1    1 1
1 1    1           1
1 1         1    1 1
1 111 111 1111 111 1
1 1         1    1 1
1 1    1    1      1
1      1    1    1 1
1 1    1         1 1
1 11 111111 111 11 1
1                  1
11111111111111111111
"""
        self.block_size = 8
        self.occupancy = np.array(
            [np.array(list(map(lambda c: 1 if c == '1' else 0, line))) for line in self.layout.splitlines()])
        self.num_pos = int(np.sum(self.occupancy == 0))
        self.Row = np.shape(self.occupancy)[0]
        self.Col = np.shape(self.occupancy)[1]
        self.obs_height = self.block_size * self.Row
        self.obs_width = self.block_size * self.Col

    def reset(self):
        super().reset()
        self.state.position_n = self.init_pos or self.state.position_n
        return self.state.to_obs


class NineroomsNorender(NineroomsBase):
    def __init__(self, max_epilen=100, init_pos=None, goal=None, seed=0):
        super().__init__(max_epilen=max_epilen, init_pos=init_pos, goal=goal, seed=seed)

    def play(self):
        print("Press esc to exit.")
        print("steps pos reward")
        cv2.imshow('img', np.flip(self.render(), -1))
        done = 0
        reward = 0
        info = {}
        while not done:
            k = cv2.waitKey(0)
            if k == 27 or k == 113:  # esc or q
                cv2.destroyAllWindows()
                return
            elif k == 0:  # up
                obs, reward, done, info = self.step(0)
            elif k == 1:  # down
                obs, reward, done, info = self.step(1)
            elif k == 2:  # left
                obs, reward, done, info = self.step(2)
            elif k == 3:  # right
                obs, reward, done, info = self.step(3)
            cv2.imshow('img', np.flip(self.render(), -1))
            step_n = self.state.current_steps
            print("%d" % step_n + ": " + "%d" % self.state.position_n + " %.1f" % reward)
        cv2.imshow('img', np.flip(self.render(), -1))
        print(info)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
