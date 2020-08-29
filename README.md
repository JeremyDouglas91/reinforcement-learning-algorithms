# reinforcement-learning-algorithms
A repository of classic reinforcement learning algorithms implemented in TensorFlow 2.

These implementations are simple (single script) and intended for experimenting with basic gym environments, or extended for more complicated tasks. I created the repository because I found it difficult to find straighforward, easy-to-interpret TF2 implementations of these algoirthms online.

I've created a docker container to make it easy to run the algoirthms without having to download tensorflow directly onto your machine.

I will add more algoritms over time.

### Algorithms (so far)
- REINFORCE 
- DQN
- A2C

### Dependancies:
- `Docker 19.03.8`

OR

- `gym==0.17.2`
- `numpy==1.18.1`
- `tensorflow==2.0.1`
- `tensorboard==2.0.2`

### Example: Running REINFORCE

1. Pull the docker image:

`docker pull jeremydd/rl:latest`

2. Set the parameters in the .ini file (e.g.):

```
[MODEL_CONFIG]
n_fc = 8
act = elu
lr =  1e-2
gamma = 0.99

[TRAINING_CONFIG]
eps_per_epoch = 10
epochs = 100
max_steps = 200
```

3. Launch the docker container in your working directory (map the docker port 6006 to the host TCP port 6006):

`docker run -it --rm -p 6006:6006 -v "$(pwd)":/wd -w /wd jeremydd/rl:latest bash`

4. Run REINFORCE, specifying the log directory (for tensorboard) and the directory of the config file (e.g.):

`python reinforce.py --config-dir=config_reinforce_cartpole.ini --log-dir=log/ --env="CartPole-v0"`

- Training information will be displayed in the terminal

5. Launch tensorboard:

`tensorboard --logdir=log/ --bind_all`

6. Navigate to _http://localhost:6006/_ to view training metrics on tensorboard:

![CP_results](https://github.com/JeremyDouglas91/reinforcement-learning-algorithms/blob/master/REINFORCE/tensorboard_example.png)
