import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorboard
import numpy as np
import gym

import datetime
import logging
import sys
import os 

class MultinomialDistribution(tf.keras.Model):
	def call(self, logits):
		# Sample a random categorical action from the given logits.
		return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
	def __init__(self, n_a, n_fc, act):
		super().__init__('mlp_policy') # model name
		self.hidden1 = kl.Dense(n_fc, activation=act)
		self.hidden2 = kl.Dense(n_fc, activation=act)

		# value head
		self.value = kl.Dense(1, name='value')

		# Unnormalized log probabilities -> 
		# multinomial distribution over actions
		self.logits = kl.Dense(n_a, name='policy_logits')
		self.dist = MultinomialDistribution() # for sampling a single action

	def call(self, obs):
		# agent observations
		x = tf.convert_to_tensor(obs)
		# Separate FC hidden layers for 
		# state-value function & policy
		hidden_logs = self.hidden1(x)
		hidden_vals = self.hidden2(x)
		return self.logits(hidden_logs), self.value(hidden_vals)

	def action_value(self, obs):
		# `predict_on_batch()` executes `call()` under the hood.
		logits, value = self.predict_on_batch(obs)
		action = self.dist.predict_on_batch(logits)
		return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2C:
	def __init__(self, env, config, log_dir):
		# Python Logging
		self.mylogger = logging.getLogger("mylogger")

		formatter = logging.Formatter('[%(levelname)s] %(message)s')

		handler = logging.StreamHandler(stream=sys.stdout)
		handler.setFormatter(formatter)
		handler.setLevel(logging.DEBUG)

		self.mylogger.addHandler(handler)
		self.mylogger.setLevel(logging.DEBUG)

		# ENVIRONMENT
		self.env = env
		self.log_dir = log_dir

		# TRAINING PARAMS
		self.epochs = config['TRAINING_CONFIG'].getint('epochs')
		self.batch_size = config['TRAINING_CONFIG'].getint('batch_size')
		self.log_step = config['TRAINING_CONFIG'].getint('log_step')

		# MODEL PARAMS
		self.n_fc = config['MODEL_CONFIG'].getint('n_fc')
		self.act = config['MODEL_CONFIG'].get('act')
		self.lr =  config['MODEL_CONFIG'].getfloat('lr')
		self.gamma = config['MODEL_CONFIG'].getfloat('gamma')
		self.value_coef = config['MODEL_CONFIG'].getfloat('value_coef')
		self.entropy_coef = config['MODEL_CONFIG'].getfloat('entropy_coef')
		
		# instantiate the model
		self.model = Model(n_a = self.env.action_space.n, 
		              n_fc = self.n_fc,
		              act = self.act
		              )

		# Compile the model
		self.model.compile(optimizer=ko.Adam(lr=self.lr),
		              loss=[self._logits_loss, self._value_loss])

		# Storage helpers for a single batch of data.
		self.actions = np.empty((self.batch_size,), dtype=np.int32)
		self.rewards, self.dones, self.values = np.empty((3, self.batch_size))
		self.observations = np.empty((self.batch_size,) + self.env.observation_space.shape)

	def run(self):
		# Training loop: collect samples, send to optimizer, repeat updates times.
		ep_rewards = [0.0]
		avg_rewards = []
		obs = self.env.reset()
		step_counter = 0

		# Tensorboard logging 
		current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		log_dir = os.path.join(self.log_dir, current_time)
		summary_writer = tf.summary.create_file_writer(log_dir)

		self.mylogger.info('Task: {}, epochs: {}, batch size: {}'.format(self.env.unwrapped.spec.id, 
																   self.epochs,
																   self.batch_size
																   ))

		for epoch in range(self.epochs):
			for step in range(self.batch_size):
				step_counter += 1

				self.observations[step] = obs.copy()
				self.actions[step], self.values[step] = self.model.action_value(obs[None, :])
				obs, self.rewards[step], self.dones[step], _ = self.env.step(self.actions[step])
				ep_rewards[-1] += self.rewards[step]

				if step_counter % self.log_step == 0:
					log_msg = 'global_step: {}, obs: {}, act: {}, reward: {}'.format(step_counter,
																					  obs, 
																					  self.actions[step], 
																					  self.rewards[step]
																					  )
					self.mylogger.info(log_msg)
					self.mylogger.info("prev episode reward: {}".format(ep_rewards[-2]))

				if self.dones[step]:
					with summary_writer.as_default():
						tf.summary.scalar('episode reward', ep_rewards[-1], step=step_counter)
					ep_rewards.append(0.0)
					obs = self.env.reset()

			_, next_value = self.model.action_value(obs[None, :])
			returns, advs = self._returns_advantages(self.rewards, self.dones, self.values, next_value)
			# A trick to input actions and advantages through same API.
			acts_and_advs = np.concatenate([self.actions[:, None], advs[:, None]], axis=-1)

			# update weights 
			losses = self.model.train_on_batch(self.observations, [acts_and_advs, returns])

			with summary_writer.as_default():
				tf.summary.scalar('policy loss', losses[1], step=step_counter)
				tf.summary.scalar('value loss', losses[2], step=step_counter)
	
	def _value_loss(self, returns, value):
		# scaled MSE between sampled returns and estimated values
		value_loss = kls.mean_squared_error(returns, value)

		return self.value_coef * value_loss

	def _logits_loss(self, actions_and_advantages, logits):
		# A trick to input actions and advantages through the same API.
		actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)
		actions = tf.cast(actions, tf.int32)

		# loss function for policy
		# from logits ensures normalisations  
		weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
		policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

		# Entropy loss can be calculated as cross-entropy of 
		# the policy over itself.
		probs = tf.nn.softmax(logits)
		entropy_loss = kls.categorical_crossentropy(probs, probs)

		return policy_loss - self.entropy_coef * entropy_loss

	def _returns_advantages(self, rewards, dones, values, next_value):
		# `next_value` is the bootstrap value estimate of the future state (critic).
		returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

		# Returns are calculated as discounted sum of future rewards.
		for t in reversed(range(rewards.shape[0])):
			returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

		# Advantages are equal to returns - baseline (value estimates in our case).
		returns = returns[:-1]
		advantages = returns - values

		return returns, advantages
