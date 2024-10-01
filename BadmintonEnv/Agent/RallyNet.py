import logging
import os
from typing import Sequence
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
import torchsde
import torch.nn.functional as F
import numpy as np
from tqdm import trange

os.environ['CUDA_LAUNCH_BLOCKING']='1'

LAUNCH_BY_BC = True

def find_first_zero_index(lst):
    try:
        index = lst.index(0)
        return index
    except ValueError:
        return -1

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out
    
class INV(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len, device):
        super(INV, self).__init__()
        self.fc = nn.Linear(input_size + input_size, hidden_size)
        self.relu = nn.ReLU()
        self.shot_embedding = nn.Embedding(shot_type_len, 8)
        self.player_embedding = nn.Embedding(player_id_len, 8)
        self.shot_type_len = shot_type_len
        self.player_id_len = player_id_len
        self.device = device
        
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        state = state.to(self.device)
        shot_types = state[:, :, 12].long().to(self.device)
        player_ids = state[:, :, 17].long().to(self.device)
        shot_embeds = self.shot_embedding(shot_types).to(self.device)
        if player_ids[0].item() != -1:
            player_embeds = self.player_embedding(player_ids).to(self.device)
        else:
            all_player_ids = torch.arange(self.player_id_len, device=state.device).long()
            average_player_embed = self.player_embedding(all_player_ids).mean(dim=0)
            batch_size, seq_len = state.shape[:2]
            player_embeds = average_player_embed.expand(batch_size, seq_len, -1).to(self.device)
        state = torch.cat((state[:, :, :12], shot_embeds, state[:, :, 13:17], player_embeds), dim=-1)
        return state

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        land_logit = self.predict_land_area(out)
        move_logit = self.predict_move_area(out)
        shot_logit = self.predict_shot_type(out)

        return land_logit, shot_logit, move_logit

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len, shot_type_len):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, shot_type_len, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, x):
        x = x.float()
        out = self.fc(x)                 # Shape: [batch_size, hidden_size]
        out = self.relu(out)
        land_logit = self.predict_land_area(out)    # Shape: [batch_size, 2]
        move_logit = self.predict_move_area(out)    # Shape: [batch_size, 2]
        shot_logit = self.predict_shot_type(out)    # Shape: [batch_size, shot_type_len]
        return land_logit, shot_logit, move_logit
    

class BC4Serve(nn.Module):
    def __init__(self, input_size, hidden_size, player_id_len):
        super(BC4Serve, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.player_embedding = nn.Embedding(player_id_len, 8)
        # Output layers for different components of the action
        self.predict_land_area = nn.Linear(hidden_size, 2, bias=False)
        self.predict_shot_type = nn.Linear(hidden_size, 3, bias=False)
        self.predict_move_area = nn.Linear(hidden_size, 2, bias=False)

    def embed_and_transform(self, state):
        player_ids = state[:, 17].long()
        player_embeds = self.player_embedding(player_ids)   # Shape: [batch_size, 8]
        # Concatenate embeddings with other state features
        # Adjust indices based on your actual state structure
        state_features = torch.cat((state[:, :17], player_embeds), dim=1)
        return state_features

    def forward(self, x):
        x = x.float()
        embedded = self.embed_and_transform(x)
        out = self.fc(embedded)
        out = self.relu(out)
        land_logit = self.predict_land_area(out)
        move_logit = self.predict_move_area(out)
        shot_logit = self.predict_shot_type(out) 
        return land_logit, shot_logit, move_logit

class RallyNet(nn.Module):
    
    sde_type = "ito"
    noise_type = "diagonal"
    
    def __init__(self, data_size, latent_size, context_size, hidden_size, target_players, player_ids_len, shot_type_len, device, id, ts):
        super(RallyNet, self).__init__()
        self.action_dim = 5
        self.state_dim = data_size - self.action_dim
        self.log_softmax = nn.LogSoftmax(dim = -1)
        self.device = device

        self.shot_type_len = shot_type_len
        self.player_ids_len = player_ids_len
        self.target_players = target_players
        self.player_id = id
        self.ts = ts

        # ================= #
        inverse_model = INV(32, 128, player_ids_len, shot_type_len, device).to(device)
        inverse_model.load_state_dict(torch.load("./BadmintonEnv/Agent/INV_weight.pth", weights_only=True))
        inverse_model.eval()
        self.inverse_model = inverse_model
        # ================= #
        
        self.action_model = MLP(latent_size, 128, player_ids_len, shot_type_len).to(device)
        
        # Encoder.
        self.encoder = Encoder(input_size=self.state_dim, hidden_size=hidden_size, output_size=context_size)
        self.qz0_net = nn.Linear(context_size, latent_size + latent_size)
        # Decoder.
        self.f_net = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )
        
        # This needs to be an element-wise function for the SDE to satisfy diagonal noise.
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Sigmoid()
                )
                for _ in range(latent_size)
            ]
        )
        self.projector = nn.Linear(latent_size, self.state_dim)
        self.pz0_mean = nn.Parameter(torch.zeros(1, latent_size))
        self.pz0_logstd = nn.Parameter(torch.zeros(1, latent_size))
        self._ctx = None

    def SDE_embed_and_transform(self, tensor_data):
        max_length, batch_size, _ = tensor_data.shape
        # Split state and action
        state = tensor_data[:, :, :18].to(self.device)
        action = tensor_data[:, :, 18:].to(self.device)
        state = self.inverse_model.embed_and_transform(state) 
        # Concatenate the state and action back together
        transformed_data = torch.cat((state, action), dim=2)
        return transformed_data

    def contextualize(self, ctx):
        self._ctx = ctx  # A tuple of tensors of sizes (T,), (T, batch_size, d).
    
    def f(self, t, y):
        ts, ctx = self._ctx
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, ctx[i]), dim=1))

    def h(self, t, y):
        return self.h_net(y)

    def g(self, t, y):  # Diagonal diffusion.
        y = torch.split(y, split_size_or_sections=1, dim=1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=1)
    
    def list_subtraction(self, p1,p2):
        point1 = p1.copy()
        point2 = p2.copy()
        v = list(map(lambda x: x[0]-x[1], zip(point1, point2)))
        return v[0], v[1]
    

    def translation4Serve(self, state, info, player): # Calculate the full states
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
        return_state_list[8], return_state_list[9] = self.list_subtraction(list(state[3]),list(state[1]))
        return_state_list[7] = (return_state_list[8]**2 + return_state_list[9]**2)**0.5
        return_state_list[10] = state[3][0]
        return_state_list[11] = state[3][1]
        
        # opponent_state
        return_state_list[12] = 0
        return_state_list[13] = state[2][0]
        return_state_list[14] = state[2][1]
        
        # the opponent's moving direction = the opponent's current location - the player's last landing location
        if info['action'][-1] != None:
            opponent_last_x = (info['action'][-1][3][0] / 177.5)
            opponent_last_y = (info['action'][-1][3][1] + 240) / 240
            return_state_list[15], return_state_list[16] = self.list_subtraction(list(state[2]), [opponent_last_x, opponent_last_y]) 
        else:
            return_state_list[15] = 0
            return_state_list[16] = 0
        
        if player != self.target_players[self.player_id]:
            if player in self.target_players:
                return_state_list[17] = self.target_players.index(player)
            else:
                return_state_list[17] = -1
        else:
            return_state_list[17] = self.player_id
    
        return torch.FloatTensor(return_state_list)
    
    def translation4RallyNet(self, state, info, step, player): # Calculate the full states
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
        
        return_state_list[1] = info['round'][-1] - 1
        return_state_list[2] = info['score'][0] 
        return_state_list[3] = info['score'][1]
        return_state_list[4] = info['score'][0] - info['score'][1] # the difference score between the player and the opponent

        # player_state
        return_state_list[5] = state[1][0]
        return_state_list[6] = state[1][1]
        
        # ball_state
        return_state_list[8], return_state_list[9] = self.list_subtraction(list(state[3]),list(state[1]))
        return_state_list[7] = (return_state_list[8]**2 + return_state_list[9]**2)**0.5
        return_state_list[10] = state[3][0]
        return_state_list[11] = state[3][1]
        
        # opponent_state
        return_state_list[12] = state[0]
        return_state_list[13] = state[2][0]
        return_state_list[14] = state[2][1]
        
        # the opponent's moving direction = the opponent's current location - the player's last landing location
        # print(info['action'])
        if info['action'][step-1] != None:
            # print("action is ", info['action'][step-1])
            opponent_last_x = (info['action'][step-1][3][0] / 177.5)
            opponent_last_y = (info['action'][step-1][3][1] + 240) / 240
            return_state_list[15], return_state_list[16] = self.list_subtraction(list(state[2]), [opponent_last_x, opponent_last_y]) 
        else:
            # print("action is None")
            return_state_list[15] = 0
            return_state_list[16] = 0
        
        if player != self.target_players[self.player_id]:
            if player in self.target_players:
                return_state_list[17] = self.target_players.index(player)
            else:
                return_state_list[17] = -1
        else:
            return_state_list[17] = self.player_id
        
        # print(return_state_list)
        return torch.FloatTensor(return_state_list).unsqueeze(0).unsqueeze(0)
    
    @torch.no_grad()
    def action(self, states, info, launch):
        raw_states = states
        shape = (int(self.ts.shape[0]), 1, 18)
        ht = torch.zeros(shape).to(self.device)
        step = info['round'][-1]-1


        if len(info['player']) > 1 and info['player'][0] == info['player'][1]:
            target_states = info['state'][1:]
            target_player = info['player'][1:]
        elif len(info['player']) == 1:
            target_states = None
        else:
            raise RuntimeError("Unexpected runtime error. Program halted.")

        if target_states != None:
            for i in range(len(target_states)):
                ht[i,:,:] = self.translation4RallyNet(target_states[i], info, i+1, target_player[i])
            ht[step,:,:] = self.translation4RallyNet(states, info, step+1, self.target_players[self.player_id])
            if step != len(target_states):
                raise RuntimeError("Current step not equal to len(target_states) + 1")
        else:
            states = self.translation4RallyNet(states, info, step+1, self.target_players[self.player_id])
            ht[step,:,:] = states

        # print(find_first_zero_index(ht[:,0,0].tolist()))

        ht = self.SDE_embed_and_transform(ht).float().to(self.device)
        ctx = self.encoder(torch.flip(ht[:,:,:self.state_dim], dims=(0,)))
        ctx = torch.flip(ctx, dims=(0,))
        self.contextualize((self.ts, ctx))
        qz0_mean, qz0_logstd = self.qz0_net(ctx[step]).chunk(chunks=2, dim=1)
        z0 = qz0_mean + qz0_logstd.exp() * torch.randn_like(qz0_mean)
        zs = torchsde.sdeint(self, z0, self.ts[step:], names={'drift': 'h'}, dt=2/int(self.ts.shape[0]), bm=None)

        land, shot, move = self.action_model(zs)
        shot_probs = torch.softmax(shot, dim=-1)[0,:,:]  # [61, 1, 12]
        output_land = land[0,:,:].tolist()[0]
        output_move = move[0,:,:].tolist()[0]

        # ================= only for serve ================= #
        if LAUNCH_BY_BC == True:
            serve_model = BC4Serve(25, 128, self.player_ids_len).to(self.device)
            serve_model.load_state_dict(torch.load("./BadmintonEnv/Agent/BC4Serve_weight.pth", weights_only=True))
            serve_model.eval()
            bc_state = self.translation4Serve(raw_states, info, self.target_players[self.player_id])
            land, bc_shot_logit, move = serve_model(bc_state.unsqueeze(0).to(self.device))
            padded_logits = torch.nn.functional.pad(bc_shot_logit, (0, self.shot_type_len - bc_shot_logit.size(1)), "constant", 0)
            shot_probs = F.softmax(padded_logits, dim=-1)
            output_land = land.tolist()[0]
            output_move = move.tolist()[0]
        # ================= only for serve ================= #
        
        if launch == True:
            mask = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.bool).to(self.device)
            masked_probs = shot_probs.masked_fill(~mask, 0.0)
            sum_probs = masked_probs.sum(dim=-1, keepdim=True)
            shot_probs = torch.where(sum_probs > 0, masked_probs / sum_probs, mask.float() / mask.sum())
        
        shot_type_dist = Categorical(shot_probs)
        shot_type = shot_type_dist.sample()
        output_shot = shot_type.unsqueeze(1).item()
        output_shot_dist = shot_probs.tolist()[0]
        

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

            mlp_action = (output_shot, raw_states[-1], tuple(output_land), tuple(output_move), normalized_tuple)
            return mlp_action
        
    