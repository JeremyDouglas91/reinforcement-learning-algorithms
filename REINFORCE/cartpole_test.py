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

