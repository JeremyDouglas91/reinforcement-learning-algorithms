import argparse
import configparser
import datetime
import gym
import logging
from memory_buffer import MemoryBuffer
import numpy as np
import os 
import sys
import tensorflow as tf

class MyModel(tf.keras.Model):
  """A simple mlp to be used for approximating Q-functions.
  """
  def __init__(self, n_states, n_fc, act, n_actions):
    super(MyModel, self).__init__()
    self.in_layer = tf.keras.layers.InputLayer(input_shape=(n_states,))
    self.h_layers = [
                    tf.keras.layers.Dense(n_fc, act, kernel_initializer = 'RandomNormal'),
                    tf.keras.layers.Dense(n_fc, act, kernel_initializer = 'RandomNormal')
                    ]
    self.out_layer = tf.keras.layers.Dense(n_actions, 'linear', kernel_initializer = 'RandomNormal')
    
  @tf.function
  def call(self, inputs):
      z = self.in_layer(inputs)
      for layer in self.h_layers:
          z = layer(z)
      output = self.out_layer(z)
      return output

class DQN:
  def __init__(self, env, config, log_dir):
    # Environment
    self.env = env

    # MODEL PARAMETERS
    self.n_fc = config['MODEL_CONFIG'].getint('n_fc')
    self.act = config['MODEL_CONFIG'].get('act')
    self.gamma = config['MODEL_CONFIG'].getfloat('gamma')
    self.lr = config['MODEL_CONFIG'].getfloat('lr')
    self.epsilon = config['MODEL_CONFIG'].getfloat('epsilon')
    self.min_epsilon = config['MODEL_CONFIG'].getfloat('min_epsilon')
    self.epsilon_decay = config['MODEL_CONFIG'].getfloat('epsilon_decay')

    # TRAINING PARAMETERS
    self.log_dir = log_dir
    self.num_eps = config['TRAINING_CONFIG'].getint('num_eps')
    self.max_steps = config['TRAINING_CONFIG'].getint('max_steps')
    self.min_exps = config['TRAINING_CONFIG'].getint('min_exps')
    self.buffer_size = config['TRAINING_CONFIG'].getint('buffer_size')
    self.batch_size = config['TRAINING_CONFIG'].getint('batch_size')
    self.copy_step = config['TRAINING_CONFIG'].getint('copy_step')

    # Python logging
    self.mylogger = logging.getLogger("mylogger")
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    self.mylogger.addHandler(handler)
    self.mylogger.setLevel(logging.DEBUG)

    # Tensorboard logging 
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tf_log_dir = os.path.join(self.log_dir, current_time)
    self.summary_writer = tf.summary.create_file_writer(tf_log_dir)

    # MODELS, OPTIMIZER & LOSS
    self.Q_main = MyModel(
                     n_states = self.env.observation_space.shape[0],
                     n_fc = self.n_fc, 
                     act = self.act,
                     n_actions = self.env.action_space.n
                     )

    self.Q_targ = MyModel(
                     n_states = self.env.observation_space.shape[0],
                     n_fc = self.n_fc, 
                     act = self.act,
                     n_actions = self.env.action_space.n
                     )

    self.optimizer = tf.keras.optimizers.Adam(self.lr)

    self.buffer = MemoryBuffer(
                          obs_dim = self.env.observation_space.shape[0], 
                          act_dim = self.env.action_space.n, 
                          size = self.buffer_size
                          )

    # Set random seed for reproducability
    seed_value = config['TRAINING_CONFIG'].getint('seed')
    np.random.seed(seed_value)

  def train(self, total_steps):
    """ Train the DQN agent on a batch of experience of size
    batch_size using an mse loss.
    """
    # sample single batch of transitions
    batch = self.buffer.sample_batch(self.batch_size)
    o = batch['obs1']
    a = batch['acts']
    r = batch['rews']
    o2 = batch['obs2']
    d = batch['done']

    # number of actions for one-hot encoding
    num_acts = self.env.action_space.n

    # Create target for bellman update
    target_action_values = np.max(self.Q_targ.predict_on_batch(o2), axis=1)
    target_action_values = np.where(d, r, r+self.gamma*target_action_values)

    # Find the MSE loss between the taget and actual Q-values for the batch 
    with tf.GradientTape() as tape:
        main_action_values = self.Q_main(np.atleast_2d(o.astype('float32')))
        selected_action_values = tf.math.reduce_sum(main_action_values * tf.one_hot(a, num_acts), axis=1)
        loss = tf.math.reduce_sum(tf.square(target_action_values - selected_action_values))

    with self.summary_writer.as_default():
        tf.summary.scalar('value loss', loss, step=total_steps)
    
    # Extract gradients from the tape and apply using the chosen optimizer
    variables = self.Q_main.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

  def sync_weights(self):
    """Copy the weights from the main Q-network to the 
    target Q-network.
    """
    variables1 = self.Q_targ.trainable_variables
    variables2 = self.Q_main.trainable_variables
    for v1, v2 in zip(variables1, variables2):
        v1.assign(v2.numpy())
  
  def run(self):
    # log initial info
    self.mylogger.info('Task: {}, num_episodes: {}'.format(self.env.unwrapped.spec.id, 
                                                           self.num_eps))

    total_steps = 0

    for ep in range(self.num_eps):
      rewards = []
      o = self.env.reset()
      self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

      # play game
      for t in range(self.max_steps):
        # take action 
        if(np.random.random() < self.epsilon):
          a = self.env.action_space.sample()
        else:
          a = np.argmax(self.Q_main.predict(np.array([o]))[0])

        # step in environment
        o2, r, d, _ = self.env.step(a)
        rewards.append(r)

        # store transition
        self.buffer.store(o, a, r, o2, d)

        # train (action replay)
        if total_steps >= self.min_exps:
          self.train(total_steps)

        if total_steps % self.copy_step == 0:
          self.sync_weights()

        # update observation
        o = o2

        # update step count 
        total_steps += 1

        if d:
          break

      # log step and rewards 
      self.mylogger.info('global step: {}, average reward: {}'.format(total_steps, sum(rewards)))
      # log reward in tensorboard
      with self.summary_writer.as_default():
        tf.summary.scalar('episode reward', sum(rewards), step=total_steps)

if __name__ == "__main__":
  # parse args 
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', required=True)
  parser.add_argument('--config-dir', required=True)
  parser.add_argument('--log-dir', required=True)
  args = parser.parse_args()

  # read config file 
  config = configparser.ConfigParser()
  config.read(args.config_dir)
  
  # create gym env and run algo 
  env = gym.make(args.env)
  dqn = DQN(env, config, args.log_dir)
  dqn.run()

