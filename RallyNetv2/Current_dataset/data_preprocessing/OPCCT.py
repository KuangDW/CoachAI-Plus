# %%
from pickle import TRUE
import pickle
import numpy as np
import pandas as pd
import os
import random
import sys
import csv
import json

match_state = ['rally', 'ball_round', 'player_score', 'opponent_score', 'score_status']
player_state = ['player_location_x', 'player_location_y']
ball_state = ['ball_distance','ball_distance_x','ball_distance_y',  'hit_x', 'hit_y']
opponent_state = ['opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y', "player"]
state_col = match_state + player_state + ball_state + opponent_state
action_col =  ['landing_x', 'landing_y', 'player_type','player_move_location_x','player_move_location_y','getpoint_player', "rally_id", "set", "match_id"]

fold_number = 0
goal = "train"

file_name = 'data_tables/' + 'total_traj_' + goal + '_' + str(fold_number) + '.pkl'
output_front_name = ''

total_rallies = pickle.load(open(file_name, 'rb'))[0]
player_rallies = []
opponent_rallies = []

terminate_ball = [0]*23

p_score = 0
o_score = 0 

for rally in total_rallies:

    player_rally = []
    opponent_rally = []

    if len(rally) >= 2:
        player_num = rally[0][17]
        opponent_num = rally[1][17]
    else:
        player_num = rally[0][17]
        opponent_num = -1
        opponent_rally.append(terminate_ball.copy())
    
    for ball in rally:
        if ball[17] == player_num:
            if np.isnan(ball[2]):
                ball[2] = p_score
                ball[3] = o_score
                ball[4] = p_score - o_score
            else:
                p_score = ball[2]
                o_score = ball[3]

            player_rally.append(ball[:23].copy())
        elif ball[17] == opponent_num:
            if np.isnan(ball[2]):
                ball[2] = o_score
                ball[3] = p_score
                ball[4] = o_score - p_score
            else:
                p_score = ball[2]
                o_score = ball[3]
                ball[2] = o_score 
                ball[3] = p_score
                ball[4] = o_score - p_score
            
            opponent_rally.append(ball[:23].copy())
        
    if len(player_rally) != 0:
        player_rallies.append(player_rally)
    else:
        print("fatal error player")
    
    if len(opponent_rally) != 0:
        opponent_rallies.append(opponent_rally)
    else:
        print("fatal error opponent")
        print(rally)


output_name = output_front_name + 'player_' + goal + '_' + str(fold_number) + '.pkl'
output = open(output_name, 'wb')
pickle.dump(player_rallies, output)
output.close()

output_name = output_front_name + 'opponent_' + goal + '_' + str(fold_number) + '.pkl'
output = open(output_name, 'wb')
pickle.dump(opponent_rallies, output)
output.close()


