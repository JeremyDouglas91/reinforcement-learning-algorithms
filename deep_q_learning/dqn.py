from memory_buffer import MemoryBuffer
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class MyModel(tf.keras.Model):
  """A simple mlp to be used for approximating Q-functions.
  """
  def __init__(self, num_states, hidden_units, hidden_activations, num_actions):
    super(MyModel, self).__init__()
    self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
    self.hidden_layers = []
    for i, n in enumerate(hidden_units):
      self.hidden_layers.append(tf.keras.layers.Dense(
          units = n, 
          activation = hidden_activations[i], 
          kernel_initializer = 'RandomNormal'))
    self.output_layer = tf.keras.layers.Dense(
        units = num_actions, 
        activation = 'linear', 
        kernel_initializer = 'RandomNormal')
    
  @tf.function
  def call(self, inputs):
      z = self.input_layer(inputs)
      for layer in self.hidden_layers:
          z = layer(z)
      output = self.output_layer(z)
      return output

class DQN:
  @staticmethod
  def train(Q_main, Q_targ, gamma, buffer, batch_size, optimizer, env):
    """ Train the DQN agent on a batch of experience of size
    batch_size using an mse loss.
    """
    # sample single batch of transitions
    batch = buffer.sample_batch(batch_size)
    o = batch['obs1']
    a = batch['acts']
    r = batch['rews']
    o2 = batch['obs2']
    d = batch['done']

    # number of actions for one-hot encoding
    num_acts = env.action_space.n

    # Create target for bellman update
    target_action_values = np.max(Q_targ.predict(o2), axis=1)
    target_action_values = np.where(d, r, r+gamma*target_action_values)

    # Find the MSE loss between the taget and actual Q-values for the batch 
    with tf.GradientTape() as tape:
        main_action_values = Q_main(np.atleast_2d(o.astype('float32')))
        selected_action_values = tf.math.reduce_sum(main_action_values * tf.one_hot(a, num_acts), axis=1)
        # selected_action_values = tf.math.reduce_sum(main_action_values * a, axis=1)
        loss = tf.math.reduce_sum(tf.square(target_action_values - selected_action_values))
    
    # Extract gradients from the tape and apply using the chosen optimizer
    variables = Q_main.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

  @staticmethod
  def sync_weights(Q_main, Q_target):
    """Copy the weights from the main Q-network to the 
    target Q-network.
    """
    variables1 = Q_target.trainable_variables
    variables2 = Q_main.trainable_variables
    for v1, v2 in zip(variables1, variables2):
        v1.assign(v2.numpy())
  
  @staticmethod
  def run(env, agent_params, training_params):
    # Set random seed for reproducability
    seed_value = training_params['seed_value']
    np.random.seed(seed_value)

    # MODEL PARAMETERS
    hidden_dims = agent_params['hidden_layer_sizes']
    hidden_activations = agent_params['hidden_layer_activations']
    gamma = agent_params['discount_factor']
    lr = agent_params['learning_rate']
    epsilon = agent_params['epsilon']
    min_epsilon = agent_params['min_epsilon']
    epsilon_decay = agent_params['epsilon_decay']

    # TRAINING PARAMETERS
    num_episodes = training_params['num_episodes']
    max_steps = training_params['max_steps']
    min_experiences = training_params['min_experiences']
    biffer_size = training_params['buffer_size']
    batch_size = training_params['batch_size']
    copy_step = training_params['copy_step']
    plot = training_params['plot_results']


    # MODELS, OPTIMIZER & LOSS
    Q_main = MyModel(
                     num_states = env.observation_space.shape[0],
                     hidden_units = hidden_dims, 
                     hidden_activations = hidden_activations,
                     num_actions = env.action_space.n
                     )

    Q_targ = MyModel(
                     num_states = env.observation_space.shape[0], 
                     hidden_units = hidden_dims, 
                     hidden_activations = hidden_activations,
                     num_actions = env.action_space.n
                     )

    optimizer = tf.keras.optimizers.Adam(lr)

    buffer = MemoryBuffer(
                          obs_dim = env.observation_space.shape[0], 
                          act_dim = env.action_space.n, 
                          size = biffer_size
                          )

    # Train agent
    all_rewards = []
    total_steps = 0

    for ep in range(num_episodes):
      o = env.reset()
      all_rewards.append(0.0)
      epsilon = max(min_epsilon, epsilon * epsilon_decay)

      # play game
      for t in range(max_steps):
        # take action 
        if(np.random.random() < epsilon):
          a = env.action_space.sample()
        else:
          a = np.argmax(Q_main.predict(np.atleast_2d(o))[0])

        # step in environment
        o2, r, d, _ = env.step(a)
        all_rewards[-1] += r

        # store transition
        buffer.store(o, a, r, o2, d)

        # train (action replay)
        if total_steps >= min_experiences:
          DQN.train(Q_main, Q_targ, gamma, buffer, batch_size, optimizer, env)

        if total_steps % copy_step == 0:
          DQN.sync_weights(Q_main, Q_targ)

        # update observation
        o = o2

        # update step count 
        total_steps += 1

        if d:
          print("episode: {}, reward: {}".format(ep, np.mean(all_rewards[-1])))
          break

    if plot:
      smooth_returns = [r if i<20 else np.mean(all_rewards[i-20:i]) for i, r in enumerate(all_rewards)]
      fig, ax = plt.subplots(figsize=(14,7))
      ax.set_title("DQN Agent | Smoothed Returns\n{}".format(env.unwrapped.spec.id))
      ax.set_xlabel("Episodes")
      ax.set_ylabel("Smoothed Returns")
      ax.plot(smooth_returns)
      plt.savefig("DQN_results.png", dpi=300)

