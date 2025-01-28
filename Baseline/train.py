import sys
import os
import argparse

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "BadmintonEnv"))
sys.path.append(project_path)
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_path)

from Environment import BadmintonEnv
from Agent.BC import BC
from Agent.RallyNet import RallyNet
from config import BaseParameters, PDQNHyperParameters, HPPOHyperParameters
import pickle
import torch

def make_env(player_name='CHOU Tien Chen', opponent_name='CHOU Tien Chen', is_constraint=False):

    with open('./input_data/target_players_ids.pkl', 'rb') as f:
        target_players = pickle.load(f)
    with open('./BadmintonEnv/Agent/hyperparameters.pkl', 'rb') as f:
        hyperparameters = pickle.load(f)

    device = BaseParameters.device

    opponent_id = target_players.index(opponent_name)
    opponent_agent = RallyNet(
        data_size= hyperparameters['data_size'], 
        latent_size=hyperparameters['latent_size'],
        context_size=hyperparameters['context_size'],
        hidden_size=hyperparameters['hidden_size'],
        player_ids_len = hyperparameters['player_ids_len'],
        target_players = hyperparameters['target_players'],
        shot_type_len = hyperparameters['shot_type_len'],
        ts = hyperparameters['ts'],
        id = opponent_id,
        device = device,
    ).to(device)
    
    opponent_agent.load_state_dict(torch.load("./BadmintonEnv/Agent/RallyNet_weight.trc", map_location = device, weights_only=True))

    env = BadmintonEnv(None, opponent_agent, 'CHOU Tien Chen', opponent_name, 10, 
                    is_match = True, is_constraint = is_constraint, is_have_serve_state = False, filepath = './output_data/output_game_1.csv')

    return env

from Baseline.Agent.PDQNAgent import PDQNAgent
from Baseline.Agent.HybridPPOAgent import HPPOAgent

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    agent_choices = ['PDQN', 'HybridPPO']

    parser.add_argument('--player', type=str, default='CHOU Tien Chen', help="oppponent player name")
    parser.add_argument('--agent', type=str, choices=agent_choices, default='PDQN', help="agent name")
    parser.add_argument('--constraint', type=bool, default=False, help="use contraint or not")

    args = parser.parse_args()

    train_env = make_env(args.player, args.player, is_constraint=args.constraint)
    eval_env = make_env(args.player, args.player, is_constraint=args.constraint)

    BaseParameters.train_env = train_env
    BaseParameters.eval_env = eval_env

    if args.agent == 'PDQN':
        agent = PDQNAgent(
            BaseParameters, PDQNHyperParameters
        )
    elif args.agent == 'HybridPPO':
        agent = HPPOAgent(
            BaseParameters, HPPOHyperParameters
        )
    else:
        raise ValueError("Invalid agent name")

    agent.train()

