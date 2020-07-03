# reinforcement-learning-algorithms
A repository of classic reinforcement learning algorithms implemented in TensorFlow 2.

These implementations are simple and intended for experimenting with basic gym environments, or extended for more complicated tasks. I will add more algoritms over time.

I have made an effort to create simple, modular python files which allow each alorithm to be run for any gym environment and configuration of parameters.

### Algorithms (so far)
- REINFORCE 
- Deep-Q Learning

### Dependancies:
- `numpy==1.18.1`
- `tensorflow-cpu==2.1.0`
- `tensorboard==2.1.1`

### Example: running reinforce

1. Set up the config file

```
[
	{
		"_comment" : "AGENT PARAMETERS",
		"hidden_layer_sizes" : [4,4], 
		"hidden_layer_activations" : ["elu", "elu"],
		"learning_rate" : 1e-2,
		"discount_factor" : 0.95
	},
	{
		"_comment" : "TRAINING PARAMETERS",
		"num_epochs" : 100,
		"episodes_per_epoch" : 10,
		"max_steps" : 1000,
		"plot_results" : true
	}
]
```

2. Create a driver file (with the envonment you want to test), `cartpole_test.py`

```
import argparse
import gym
import json
from reinforce import reinforce

def main(config_path):
	with open(config_path, 'r') as f:
		config_dict = json.load(f)
	agent_params = config_dict[0]
	training_params = config_dict[1]
	env = gym.make("CartPole-v0")
	reinforce.run(env, agent_params, training_params)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_path', required=True)
	args = parser.parse_args()
	main(args.config_path)
```

3. Run `cartpole_test.py` passing it the path to your config file

`$ python cartpole_test.py --config_path "cartpole_config.json" `

4. Rewards will be displayed in your terminal:
```
epoch: 0, average reward: 38.2
epoch: 1, average reward: 37.2
epoch: 2, average reward: 36.4
epoch: 3, average reward: 33.5
epoch: 4, average reward: 57.0
epoch: 5, average reward: 26.4
epoch: 6, average reward: 57.4
epoch: 7, average reward: 42.9
epoch: 8, average reward: 68.0
epoch: 9, average reward: 48.9
epoch: 10, average reward: 77.6
epoch: 11, average reward: 59.8
epoch: 12, average reward: 91.6
epoch: 13, average reward: 63.6
epoch: 14, average reward: 109.4
```
5. Final results will be plotted and saved to the working directory

![CP_results](https://github.com/JeremyDouglas91/reinforcement-learning-algorithms/blob/master/REINFORCE/CartPole-v0_returns.png)
