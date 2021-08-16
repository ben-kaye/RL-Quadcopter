from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from numpy import random
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import simfunc

import math

# TODO: implement simfunc as PyEnvironment


class QCEnv(py_environment.PyEnvironment):
    def __init__(self):

        self.dtype = np.float32

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=self.dtype, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(12,), dtype=self.dtype, minimum=-np.inf, name='observation')  # should min be inf?

        RNG = np.random.default_rng()
        random_pose = [u*np.pi/3 for u in RNG.standard_normal(3)]

        self._state = np.array([0 for i in range(12)], dtype=self.dtype)

        self.target_pose = random_pose

        self.sim_time = 30
        self.dt = 1e-3
        self.time = float(0)

        self._episode_ended = False

    def reward_func(state, target):
        e_angles = state[6:9] # TODO convert representation to quaternions
        return -math.log(1 + float((target - e_angles).dot(target - e_angles)))

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self.time = 0
        self._episode_ended = False
        return ts.restart(np.array([self._state], dtype=self.dtype))

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        simfunc.state_advance(self._state, action, self.dt)

        self.time += self.dt

        if self.time > self.sim_time:
            self._episode_ended = True

        # might convert to quaternion instead for uniqueness
        new_reward = self.reward_func(self._state, self.target_pose)

        if self._episode_ended:
            return ts.termination(np.array([self._state], dtype=self.dtype), reward=new_reward)
        else:
            return ts.transition(np.array([self._state], dtype=self.dtype), reward=new_reward)

    