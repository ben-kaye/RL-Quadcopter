from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
from threading import active_count

from tensorflow._api.v2.compat.v1 import train
import numpy as np
# import PIL.Image
# import pyvirtualdisplay
# import imageio
# import IPython
import matplotlib.pyplot as plt
import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

import episode_funcs as epfun

import QC_env



train_env = tf_py_environment.TFPyEnvironment(QC_env.QCEnv)
eval_env = tf_py_environment.TFPyEnvironment(QC_env.QCEnv)

num_iterations = 250 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000 # @param {type:"integer"}

learning_rate = 1e-3 # @param {type:"number"}
log_interval = 25 # @param {type:"integer"}
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 50 # @param {type:"integer"}

tf.compat.v1.enable_v2_behavior()

layer_params = (200, 200, 200) # 3 layer deep NN

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params = layer_params
)

optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.compat.v2.Variable(0)

tf_agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimiser,
    normalize_returns=True,
    train_step_counter=train_step_counter)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec = tf_agent.collect_data_spec,
    batch_size = train_env.batch_size,
    max_length = replay_buffer_capacity)

# try: # not sure what this does?
#     %%time
# except:
#     pass

tf_agent.train = common.function(tf_agent.train)

tf_agent.train_step_counter.assign(0)
avg_return = epfun.compute_avg_return(train_env, tf_agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):
    epfun.collect_episode(train_env, tf_agent.collect_policy, replay_buffer, collect_episodes_per_iteration)
    experience = replay_buffer.gather_all()
    train_loss = tf_agent.train(experience)

    replay_buffer.clear()
    step = tf_agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        avg_return = epfun.compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: average return = {1}'.format(step, train_loss))
        returns.append(avg_return)


#%%
steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('average return')
plt.xlabel('step')

print('exec done')