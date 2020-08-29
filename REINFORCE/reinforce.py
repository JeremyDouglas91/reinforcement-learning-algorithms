import argparse
import configparser
import datetime
import gym
import logging
import tensorflow as tf
import numpy as np
import os
import sys

tf.keras.backend.set_floatx('float64')

class REINFORCE: 
  """
  A simple implementation of the REINFORCE (Policy Gradient) reinforcement 
  learning algorithm. 
  """
  def __init__(self, env, config, log_dir):
    """
    parameters
    ----------
    env: gym environment
    config: config object containing parameter values
    log_dir: directory to save tensorboard logs 
    """
    # Gym environment
    self.env = env

    # Model parameters
    self.n_fc = config['MODEL_CONFIG'].getint('n_fc')
    self.act = config['MODEL_CONFIG'].get('act')
    self.lr =  config['MODEL_CONFIG'].getfloat('lr')
    self.gamma = config['MODEL_CONFIG'].getfloat('gamma')

    # Training parameters
    self.num_epochs = config['TRAINING_CONFIG'].getint('epochs')
    self.ep_per_epoch = config['TRAINING_CONFIG'].getint('eps_per_epoch')
    self.max_steps = config['TRAINING_CONFIG'].getint('max_steps')
    self.log_dir = log_dir

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

    # Create policy network 
    self.policy = tf.keras.models.Sequential()
    self.policy.add(tf.keras.layers.Input(shape = self.env.observation_space.shape))
    self.policy.add(tf.keras.layers.Dense(self.n_fc, activation=self.act))
    self.policy.add(tf.keras.layers.Dense(self.n_fc, activation=self.act))
    self.policy.add(tf.keras.layers.Dense(self.env.action_space.n, activation='softmax')) # logits 

    # loss function & optimizer
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
    self.ce_loss = tf.keras.losses.CategoricalCrossentropy()


  def sample_action(self, o, training):
    """
    Samples a single action from the stochastic policy, and
    in addition returns the x-entropy loss. 

    parameters
    ----------
    o: a single observation vector
    training: boolean

    returns
    -------
    action: sampled from stochastic policy
    loss: x-entropy loss (to be used for gradient computation)
    """
    o = tf.reshape(o, (1,-1))
    output = self.policy(o, training=training)
    action = tf.random.categorical(tf.math.log(output), num_samples = 1)
    label = tf.reshape(tf.one_hot(action, depth=self.env.action_space.n), shape=(1,-1))
    loss = self.ce_loss(label, output)
    return action, loss

  def discount_rewards(self, rewards):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
      cumulative_rewards = rewards[step] + cumulative_rewards * self.gamma
      discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

  def discount_and_normalize_rewards(self, all_rewards):
    """
    Discounts and normalises rewards collected for wach 
    trajectory in the epoch.

    parameters
    ----------
    all_rewards: reward vectors - one per epoch.

    returns
    -------
    norm_rewards: normalised and discounted reward vector.
    """
    all_discounted_rewards = [self.discount_rewards(rewards) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    norm_rewards = [(discounted_rewards - reward_mean)/reward_std 
                    for discounted_rewards in all_discounted_rewards]
    return norm_rewards
  
  def run(self):
    """
    Run the REINFORCE algoirthm.
    """
    # log initial info
    self.mylogger.info('Task: {}, epochs: {}, episodes per epoch: {}'.format(self.env.unwrapped.spec.id, 
                                                                             self.num_epochs,
                                                                             self.ep_per_epoch))

    # global step counter
    step_counter = 0

    for epoch in range(self.num_epochs):
      all_gradients = []
      all_rewards = []

      for episode in range(self.ep_per_epoch):
        gradients = []
        rewards = []
        o, r, done, _ = self.env.reset(), 0, False, {}

        for step in range(self.max_steps):
          # sample action from policy & collect gradient
          with tf.GradientTape() as tape:
            a, loss_value = self.sample_action(o, training=True)
          grads = tape.gradient(loss_value, self.policy.trainable_variables)

          # step in environment
          o, r, done, _ = self.env.step(a.numpy()[0][0])

          # collect grads and rewards, update step count
          gradients.append(grads)
          rewards.append(r) 
          step_counter += 1 

          # log loss in tensorboard
          with self.summary_writer.as_default():
            tf.summary.scalar('policy loss', loss_value, step=step_counter)
          
          # break if state is terminal 
          if done:
            break

        all_gradients.append(gradients)
        all_rewards.append(rewards)

        # log reward in tensorboard
        with self.summary_writer.as_default():
            tf.summary.scalar('episode reward', sum(rewards), step=step_counter)

      # log training progress
      average_reward = np.mean([sum(rewards) for rewards in all_rewards])
      self.mylogger.info('global step: {}, epoch: {}, average reward: {}'.format(step_counter, 
                                                                                 epoch, 
                                                                                 average_reward))

      # compute discounted & normalised returns
      all_rewards = self.discount_and_normalize_rewards(all_rewards)
      
      # update policy network
      grads_to_apply = []
      for var_index in range(len(self.policy.get_weights())):
        mean_gradients = np.mean(
                                [reward * all_gradients[game_index][step][var_index]
                                for game_index, rewards in enumerate(all_rewards)
                                for step, reward in enumerate(rewards)], axis=0
                                )
        grads_to_apply.append(mean_gradients)
      self.optimizer.apply_gradients(zip(grads_to_apply, self.policy.trainable_variables))

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
  reinforce = REINFORCE(env, config, args.log_dir)
  reinforce.run()
  
