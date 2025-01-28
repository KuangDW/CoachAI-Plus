import os
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter


class BaseAgent(object):
    def __init__(self, config):

        self.seed = config.seed
        self.set_random_seeds(self.seed)
        self.eval_interval = config.eval_interval
        self.evaluate_rallies = config.evaluate_rallies
        
        self.train_env = config.train_env
        self.eval_env = config.eval_env

        self.device = config.device
        self.log_dir = config.log_dir

        self.writer = SummaryWriter(self.log_dir)


    def act(self, state, eval=False):
        """
        Determines the action to take in the given state.
        :param state:
        :param eval:
        :return:
        """
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
    
    def set_random_seeds(self, random_seed):
        """
        Sets all possible random seeds to results can be reproduces.

        :param random_seed:
        :return:
        """
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)

    def encode_positions(self, my_pos, opp_pos, ball, land, move):
        """
        Encodes the action to a tensor.
        :param action:
        :return:
        """
        my_pos = list(my_pos)
        opp_pos = list(opp_pos)
        ball = list(ball)
        land = list(land)
        move = list(move)
        my_pos[0] = my_pos[0] / 177.5
        my_pos[1] = (my_pos[1] + 240) / 240
        opp_pos[0] = opp_pos[0] / 177.5
        opp_pos[1] = -(opp_pos[1] - 240) / 240
        ball[0] = ball[0] / 177.5
        ball[1] = (ball[1] + 240) / 240
        land[0] = land[0] / 177.5
        land[1] = -(land[1] - 240) / 240
        move[0] = move[0] / 177.5
        move[1] = (move[1] + 240) / 240
        return tuple(my_pos), tuple(opp_pos), tuple(ball), tuple(land), tuple(move)
    
    def decode_positions(self, land, move):
        """
        Encodes the action to a tensor.
        :param action:
        :return:
        """
        land = list(land)
        move = list(move)

        land[0] = land[0]*177.5
        land[1] = -(land[1]*240) + 240
        move[0] = move[0]*177.5
        move[1] = (move[1]*240) - 240
        return tuple(land), tuple(move)