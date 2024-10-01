from Environment import BadmintonEnv
from Agent.BC import BC
from Agent.RallyNet import RallyNet
import pickle
import torch

# ================ Load real player names list =============== #
with open('./input_data/target_players_ids.pkl', 'rb') as f:
    target_players = pickle.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ================ Specify the player name and load your agent (we provide the Behavior Cloning Model here) =============== #
player_name = 'CHOU Tien Chen'
player_id = target_players.index(player_name)
player_agent = BC(32, 128, len(target_players), 12, player_id)
player_agent.load_state_dict(torch.load("./BadmintonEnv/Agent/BC_weight.pth", weights_only=True))
player_agent.eval()

# ================ Or use our proposed RallyNetv2 as player  =============== #
# player_name = 'CHOU Tien Chen'
# player_id = target_players.index(player_name)
# with open('./BadmintonEnv/Agent/hyperparameters.pkl', 'rb') as f:
#     hyperparameters = pickle.load(f)
# player_agent = RallyNet(
#         data_size= hyperparameters['data_size'], 
#         latent_size=hyperparameters['latent_size'],
#         context_size=hyperparameters['context_size'],
#         hidden_size=hyperparameters['hidden_size'],
#         player_ids_len = hyperparameters['player_ids_len'],
#         target_players = hyperparameters['target_players'],
#         shot_type_len = hyperparameters['shot_type_len'],
#         ts = hyperparameters['ts'],
#         id = player_id,
#         device = device,
#     ).to(device)
# player_agent.load_state_dict(torch.load("./BadmintonEnv/Agent/RallyNet_weight.trc", map_location = device, weights_only=True))


# ================ Specify the real opponent AI using RallyNet (the state-of-the-art imitation learning model) =============== #
opponent_name = 'Kento MOMOTA'
with open('./BadmintonEnv/Agent/hyperparameters.pkl', 'rb') as f:
    hyperparameters = pickle.load(f)
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

# ================ Specify rally length =============== #
rallies = 5

# ================ Construct the environment for interaction =============== #
env = BadmintonEnv(player_agent, opponent_agent, player_name, opponent_name, rallies, 
                   is_match = True, is_constraint = True, is_have_serve_state = False, filepath = './output_data/output_game_1.csv')

# ================ Competition starts =============== #
for rally in range(1, rallies + 1):
    print("rally :", rally)
    states, info, done, launch = env.reset()
    while not done :
        action = player_agent.action(states, info, launch)
        states, reward, info, done, launch = env.step(action, launch)
    print("score: ", info['env_score'])
    print("round: ", info['round'][-1]-1)
    print()
# ================ Close the environment and save the result =============== #
env.close()
