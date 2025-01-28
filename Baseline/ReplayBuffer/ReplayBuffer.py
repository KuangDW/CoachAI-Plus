import numpy as np
import torch
from collections import deque
import random

class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action_type, action_land, action_move, reward, next_state, done):

        action_type = [action_type]
        action_param = list(action_land) + list(action_move)
        reward = [reward]
        done = [1 if done else 0]

        self.buffer.append(
            (state, action_type, action_param, reward, next_state, done)
        )

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        transitions = random.sample(self.buffer, batch_size)
        return (np.asarray(x) for x in zip(*transitions))
    

class GaeReplayBuffer:

    class Trajectory:
        def __init__(self):
            self.transitions = {
                "state": [],
                "action_type": [],
                "action_param": [],
                "reward": [],
                "value": [],
                "discrete_logp": [],
                "continous_logp": [],
                "done": []
            }
        def __len__(self):
            return len(self.transitions["state"])
        
        def clear(self):
            for key in self.transitions.keys():
                self.transitions[key].clear()

        def append(self, state, action_type, action_land, action_move, reward, value, discrete_logp, continous_logp, done):
            self.transitions["state"].append(state)
            self.transitions["action_type"].append(action_type)
            self.transitions["action_param"].append(list(action_land) + list(action_move))
            self.transitions["reward"].append(reward)
            self.transitions["value"].append(value)
            self.transitions["discrete_logp"].append(discrete_logp)
            self.transitions["continous_logp"].append(continous_logp)
            self.transitions["done"].append(done)


    def __init__(self, horizon, batch_size, gamma, gae_lambda):
        self.horizon = horizon
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.trajectories = []
        self.batch_index = 0

    def __len__(self):
        sample_count = 0
        for trajectory in self.trajectories:
            sample_count += len(trajectory)
        return sample_count

    def append(self, sample):
        self.trajectories.append(sample)

    def get_batch(self):
        pass

    def compute_gae(self):

        for trajectory in self.trajectories:

            values = trajectory.transitions["value"]
            rewards = trajectory.transitions["reward"]
            dones = trajectory.transitions["done"]

            if dones[-1]:
                values.append(0)
            else:
                values.append(values[-1])
            gae = 0
            returns = []
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
                returns.insert(0, gae + values[i])

            trajectory.transitions["return"] = returns

    def clear(self):
        self.trajectories.clear()
        self.batch_index = 0