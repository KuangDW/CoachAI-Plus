import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

def list_subtraction(p1,p2):
    point1 = p1.copy()
    point2 = p2.copy()
    v = list(map(lambda x: x[0]-x[1], zip(point1, point2)))
    return v[0], v[1]


# Define the MLP model
class BC(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len, player_id):
        super(BC, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.player_id = player_id
        # Embeddings for categorical features within the state
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        
        # Output layers for different components of the action
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        # Assuming state[:, 12] is shot_type and state[:, 17] is player_id
        shot_types = state[:, 12].long()
        player_ids = state[:, 17].long()
        shot_embeds = self.shot_embedding(shot_types)       # Shape: [batch_size, 8]
        player_embeds = self.player_embedding(player_ids)   # Shape: [batch_size, 8]
        
        # Concatenate embeddings with other state features
        # Adjust indices based on your actual state structure
        state_features = torch.cat((state[:, :12], shot_embeds, state[:, 13:17], player_embeds), dim=1)  # Shape: [batch_size, 32]
        return state_features

    def forward(self, x):
        x = x.float()
        embedded = self.embed_and_transform(x)  # Shape: [batch_size, 32]
        out = self.fc(embedded)                 # Shape: [batch_size, hidden_size]
        out = self.relu(out)

        land_logit = self.predict_land_area(out)    # Shape: [batch_size, 2]
        move_logit = self.predict_move_area(out)    # Shape: [batch_size, 2]
        shot_logit = self.predict_shot_type(out)    # Shape: [batch_size, shot_type_len]

        return land_logit, shot_logit, move_logit
    
    def translation(self, state, info): # Calculate the full states
        return_state_list = [0]*18
        
        state = list(state)
        state[1] = list(state[1])
        state[2] = list(state[2])
        state[3] = list(state[3])

        state[1][0] = state[1][0] / 177.5
        state[1][1] = - (state[1][1] + 240) / 240

        state[2][0] = - (state[2][0] / 177.5)
        state[2][1] = - (state[2][1] - 240) / 240

        state[3][0] = state[3][0] / 177.5
        state[3][1] = - (state[3][1] + 240) / 240
        
        # match_state = ['rally', 'ball_round', 'player_score', 'opponent_score', 'score_status']
        # player_state = ['player_location_x', 'player_location_y']
        # ball_state = ['ball_distance','ball_distance_x','ball_distance_y',  'hit_x', 'hit_y']
        # opponent_state = ['opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y']

        # match_state
        if info['rally'] < 0:
            return_state_list[0] = 1
        else:
            return_state_list[0] = info['rally']
        
        return_state_list[1] = info['round'][-1]
        return_state_list[2] = info['score'][0] 
        return_state_list[3] = info['score'][1]
        return_state_list[4] = info['score'][0] - info['score'][1] # the difference score between the player and the opponent

        # player_state
        return_state_list[5] = state[1][0]
        return_state_list[6] = state[1][1]
        
        # ball_state
        return_state_list[8], return_state_list[9] = list_subtraction(list(state[3]),list(state[1]))
        return_state_list[7] = (return_state_list[8]**2 + return_state_list[9]**2)**0.5
        return_state_list[10] = state[3][0]
        return_state_list[11] = state[3][1]
        
        # opponent_state
        return_state_list[12] = state[0]
        return_state_list[13] = state[2][0]
        return_state_list[14] = state[2][1]
        
        # the opponent's moving direction = the opponent's current location - the player's last landing location
        if info['action'][-1] != None:
            opponent_last_x = (info['action'][-1][3][0] / 177.5)
            opponent_last_y = (info['action'][-1][3][1] + 240) / 240
            return_state_list[15], return_state_list[16] = list_subtraction(list(state[2]), [opponent_last_x, opponent_last_y]) 
        else:
            return_state_list[15] = 0
            return_state_list[16] = 0
        
        return_state_list[17] = self.player_id
    
        return torch.FloatTensor(return_state_list)
    
    def action(self, state, info, launch):
        bc_state = self.translation(state, info)
        bc_land, bc_shot_logit, bc_move = self.forward(bc_state.unsqueeze(0))
        bc_shot_type_probs = F.softmax(bc_shot_logit, dim=-1)
        

        if launch == True:
            mask = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool)
            masked_probs = bc_shot_type_probs.masked_fill(~mask, 0.0)
            sum_probs = masked_probs.sum(dim=-1, keepdim=True)
            bc_shot_type_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())
        
        bc_shot_type_dist = Categorical(bc_shot_type_probs)
        bc_shot_type = bc_shot_type_dist.sample()
        output_shot = bc_shot_type.unsqueeze(1).item()
        output_shot_dist = bc_shot_type_probs.tolist()[0]
        output_land = bc_land.tolist()[0]
        output_move = bc_move.tolist()[0]

        if output_shot == 11:
            return None
        else:
            prob_array = np.array(output_shot_dist)
            prob_array = prob_array[1:11]
            normalized_array = prob_array / prob_array.sum()
            normalized_tuple = tuple(normalized_array)

            output_land[0] = -(output_land[0]*177.5)
            output_land[1] = -(output_land[1]*240) + 240
            output_move[0] = output_move[0]*177.5
            output_move[1] = (output_move[1]*240) - 240

            bc_action = (output_shot, state[-1], tuple(output_land), tuple(output_move), normalized_tuple)
            return bc_action
