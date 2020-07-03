import argparse
import configparser
import gym
from a2c import A2C

def main(config_dir, log_dir):
	config = configparser.ConfigParser()
	config.read(config_dir)
	env = gym.make("CartPole-v0")
	a2c = A2C(env, config, log_dir)
	a2c.run()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config-dir', required=True)
	parser.add_argument('--log-dir', required=True)
	args = parser.parse_args()
	main(args.config_dir, args.log_dir)

