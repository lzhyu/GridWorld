import os
import threading

import gym
import numpy as np
from ..utils.test_util import *
from .fourrooms_coin import FourroomsCoinWhiteBackground
from ..utils.wrapper.wrappers import ImageInputWarpper
from stable_baselines3.common.vec_env import SubprocVecEnv,DummyVecEnv
ATARI_ENV = 2
GRID_ENV = 1


class Atari(gym.Env):

  #LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(64, 64), grayscale=False, noops=30,
      life_done=False, sticky_actions=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    #with self.LOCK:
    env = gym.envs.atari.AtariEnv(
        game=name, obs_type='image', frameskip=1,
        repeat_action_probability=0.25 if sticky_actions else 0.0,
        full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._env = env
    self._grayscale = grayscale

  @property
  def observation_space(self):
      return self._env.observation_space
    # return gym.spaces.Dict({
    #     'image': self._env.observation_space,
    #     'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
    # })

  @property
  def action_space(self):
      return self._env.action_space
    #return gym.spaces.Dict({'action': self._env.action_space})

  def close(self):
    return self._env.close()

  def reset(self):
    #with self.LOCK:
    image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs['image']

  def step(self, action): 
    action = action
    image, reward, done, info = self._env.step(action)
    if self._grayscale:
      image = image[..., None]
    obs = {'image': image, 'ram': self._env.env._get_ram()}
    return obs['image'], reward, done, info

  def render(self, mode='rgb_array'):
    return self._env.render(mode)

class SequentialEnv(gym.Env):
    def __init__(self):
        self.grid_env = ImageInputWarpper(FourroomsCoinWhiteBackground(num_coins = 1))
        self.atari_env = Atari('pong')
        self.stage = 0

    def step(self, action):
        if self.stage == GRID_ENV:
            if action == 4 or action == 5:
                action = 3
            obs, reward, done, info = self.grid_env.step(action)
            if done and self.grid_env.state.current_steps < self.grid_env.max_epilen-5:
                #switch env
                reward = 30
                self.stage = ATARI_ENV
                obs = self.atari_env.reset()
                return obs, reward, False, info
            
        elif self.stage == ATARI_ENV:
            obs, reward, done, info = self.atari_env.step(action)
        elif self.stage == 0:
            raise ValueError("env should be reset")
        return obs, reward, done, info

    def reset(self):
        self.stage = GRID_ENV
        obs = self.grid_env.reset()
        return obs
        

    @property
    def action_space(self):
        return gym.spaces.Discrete(6)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(64,64,3), dtype=np.uint8)

    def render(self):
        if self.stage==GRID_ENV:
            return self.grid_env.render()
        elif self.state==ATARI_ENV:
            return self.atari_env.render()

if __name__=='__main__':
    env = SequentialEnv()
    check_render(env)
    check_run(env)
    print(env.observation_space)
    print(env.action_space)
    print("basic check finished")
    from stable_baselines3 import A2C, PPO, TD3, SAC
    env = Atari('pong')
    kwargs = dict(n_epochs=10, n_steps=64,learning_rate=0.0001, ent_coef=0.003)
    print("pid:\t{}".format(os.getpid()))
    print(kwargs)
    model =PPO('CnnPolicy', env, verbose=1,tensorboard_log='./sequential_log/'+'test_pong'+'/',\
    **kwargs)
    model.learn(2e6)

