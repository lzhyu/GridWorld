import gym
from gym import core, spaces
import numpy as np

def normalization(data):
    data = data.astype(np.float)
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
        
class ImageInputWarpper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        screen_height = self.env.obs_height
        screen_width = self.env.obs_width
        self.observation_space = spaces.Box(low=0, high=255, shape=\
        (screen_height, screen_width,3), dtype=np.uint8)
        self.mean_obs = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        obs = self.env.render()
        info['s_tp1'] = state
        return obs.astype(np.uint8), reward, done, info

    def reset(self):
        self.env.reset()
        obs = self.env.render()
        return obs.astype(np.uint8)

class ReduceImageWarpper(gym.Wrapper):
    #The value of each pixel is subtracted from its adjacent pixel below 
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        screen_height = self.env.obs_height
        screen_width = self.env.obs_width
        self.observation_space = spaces.Box(low=0, high=255, shape=\
        (screen_height, screen_width,3), dtype=np.uint8)
        

    def reduce_obs(self, obs):
        obs[1:]=obs[1:]-obs[:-1]
        return (normalization(obs)*255).astype(np.uint8)

    def recover_obs(self, obs):
        for i in range(self.env.obs_height):
            if i==0:
                continue
            else:
                obs[i] = obs[i]+obs[i-1]
        return obs

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        obs = self.env.render()
        info['s_tp1'] = state
        
        obs = self.reduce_obs(obs)
        return obs.astype(np.int), reward, done, info

    def reset(self):
        self.env.reset()
        obs = self.env.render()
        obs = self.reduce_obs(obs)
        return obs.astype(np.int)