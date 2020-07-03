import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

class reinforce:
  @staticmethod
  def run(env, agent_params, training_params):
    # MODEL PARAMETERS
    hidden_layer_sizes = agent_params["hidden_layer_sizes"]
    hidden_layer_activations = agent_params["hidden_layer_activations"]
    lr = agent_params["learning_rate"]
    gamma = agent_params["discount_factor"]

    # TRAINING PARAMETERS
    num_epochs = training_params["num_epochs"]
    ep_per_epoch = training_params["episodes_per_epoch"]
    max_steps = training_params["max_steps"]
    plot = training_params["plot_results"] 

    # Create policy network 
    policy = tf.keras.models.Sequential()
    policy.add(tf.keras.layers.Input(shape = env.observation_space.shape))
    for n, act in zip(hidden_layer_sizes, hidden_layer_activations):
      policy.add(tf.keras.layers.Dense(n, activation=act))
    policy.add(tf.keras.layers.Dense(env.action_space.n, activation='softmax')) # logits 

    # loss function & optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    ce_loss = tf.keras.losses.CategoricalCrossentropy()

    def loss(policy, o, training):
      o = tf.reshape(o, (1,-1))
      output = policy(o, training=training)
      action = tf.random.categorical(tf.math.log(output), num_samples = 1)
      label = tf.reshape(tf.one_hot(action, depth=env.action_space.n), shape=(1,-1))
      return action, ce_loss(label, output)

    def discount_rewards(rewards, gamma):
     discounted_rewards = np.empty(len(rewards))
     cumulative_rewards = 0
     for step in reversed(range(len(rewards))):
      cumulative_rewards = rewards[step] + cumulative_rewards * gamma
      discounted_rewards[step] = cumulative_rewards
     return discounted_rewards

    def discount_and_normalize_rewards(all_rewards, gamma):
     all_discounted_rewards = [discount_rewards(rewards, gamma) for rewards in all_rewards]
     flat_rewards = np.concatenate(all_discounted_rewards)
     reward_mean = flat_rewards.mean()
     reward_std = flat_rewards.std()
     return [(discounted_rewards - reward_mean)/reward_std 
             for discounted_rewards in all_discounted_rewards] # normalisation
    
    # run training
    average_rewards = []

    for epoch in range(num_epochs):
      all_gradients = []
      all_rewards = []

      for episode in range(ep_per_epoch):
        gradients = []
        rewards = []
        o, r, done, _ = env.reset(), 0, False, {}

        for step in range(max_steps):
          # sample action from policy & collect gradient
          with tf.GradientTape() as tape:
            a, loss_value = loss(policy, o, training=True)
          grads = tape.gradient(loss_value, policy.trainable_variables)

          # step in environment
          o, r, done, _ = env.step(a.numpy()[0][0])

          # collect grads and rewards
          gradients.append(grads)
          rewards.append(r) 

          # break if done == True
          if done:
            break

        all_gradients.append(gradients)
        all_rewards.append(rewards)

      # print progress
      average_reward = np.mean([sum(rewards) for rewards in all_rewards])
      average_rewards.append(average_reward)
      print("epoch: {}, average reward: {}".format(epoch, average_reward))

      # compute discounted & normalised returns
      all_rewards = discount_and_normalize_rewards(all_rewards, gamma)
      
      # update policy network
      grads_to_apply = []
      for var_index in range(len(policy.get_weights())):
        mean_gradients = np.mean(
                                [reward * all_gradients[game_index][step][var_index]
                                for game_index, rewards in enumerate(all_rewards)
                                for step, reward in enumerate(rewards)],
                                axis=0
                                )
        grads_to_apply.append(mean_gradients)
      optimizer.apply_gradients(zip(grads_to_apply, policy.trainable_variables))

    if plot:
      fig, ax = plt.subplots(figsize=(10,6))
      ax.plot(average_rewards)
      ax.set_title("REINFORCE Agent | Return Per Epoch\n{}".format(env.unwrapped.spec.id))
      ax.set_xlabel("Epochs ({} episodes per epoch)".format(ep_per_epoch))
      ax.set_ylabel("average return over epoch")
      plt.savefig("{}_results.png".format(env.unwrapped.spec.id), dpi=300)
      plt.show()

