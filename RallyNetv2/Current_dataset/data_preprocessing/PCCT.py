
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

train_or_test = "train" # str(sys.argv[1])
fold_number = 0 # int(sys.argv[2])

if train_or_test == "raw_data" and fold_number == -1:
    output_name = 'data_tables/total_traj_all.pkl'
else:
    output_name = 'data_tables/' + 'total_traj_' + train_or_test + '_' + str(fold_number) + '.pkl'

all_player_trajectories_data = {}
player_trajectories_data = []
player_trajectory_data = []
player_state_action_pair = []
player_state_list = []
player_action_list = []

terminal_state = [0] * 17 # 17/12

# 連續型欄位
match_state = ['rally', 'ball_round', 'player_score', 'opponent_score', 'score_status']
player_state = ['player_location_x', 'player_location_y']
ball_state = ['ball_distance','ball_distance_x','ball_distance_y',  'hit_x', 'hit_y']
opponent_state = ['opponent_type', 'opponent_location_x', 'opponent_location_y', 'opponent_move_x', 'opponent_move_y', "player"]
state_col = match_state + player_state + ball_state + opponent_state
action_col =  ['landing_x', 'landing_y', 'player_type','moving_x','moving_y','getpoint_player', "rally_id", "set", "match_id"]


state_action_dividing = len(state_col)

# target_players = ['CHOU Tien Chen', 'LOH Kean Yew', 'LIN Chun Yi', 'Kodai NARAOKA']
# target_players = ['Kento MOMOTA', 'CHOU Tien Chen', 'Anthony Sinisuka GINTING', 'CHEN Long', 'CHEN Yufei', 'TAI Tzu Ying', 'Viktor AXELSEN', 'Anders ANTONSEN', 'PUSARLA V. Sindhu', 'WANG Tzu Wei', 'Khosit PHETPRADAB', 'Jonatan CHRISTIE', 'NG Ka Long Angus', 'SHI Yuqi', 'Ratchanok INTANON', 'An Se Young', 'Busanan ONGBAMRUNGPHAN', 'Mia BLICHFELDT', 'LEE Zii Jia', 'LEE Cheuk Yiu', 'Rasmus GEMKE', 'Michelle LI', 'Supanida KATETHONG', 'Carolina MARIN', 'Pornpawee CHOCHUWONG', 'Sameer VERMA', 'Neslihan YIGIT', 'Hans-Kristian Solberg VITTINGHUS', 'LIEW Daren', 'Evgeniya KOSETSKAYA', 'KIDAMBI Srikanth', 'Soniia CHEAH', 'Gregoria Mariska TUNJUNG', 'Akane YAMAGUCHI', 'HE Bingjiao', '胡佑齊', '張允澤', '許喆宇', '陳政佑', '林祐賢', '李佳豪', 'LOH Kean Yew', 'Lakshya SEN', 'Kunlavut VITIDSARN', 'WANG Hong Yang', 'Kodai NARAOKA', 'JEON Hyeok Jin', 'Wen Chi Hsu', 'Nozomi Okuhara', 'WANG Zhi Yi', 'PRANNOY H. S.', 'Chico Aura DWI WARDOYO', 'LU Guang Zu', 'ZHAO Jun Peng', 'Kenta NISHIMOTO', 'NG Tze Yong', 'Victor SVENDSEN', 'WENG Hong Yang', 'Aakarshi KASHYAP', 'LI Shi Feng', 'KIM Ga Eun', 'HAN Yue', 'Other', 'NYCU', 'Kanta TSUNEYAMA', 'YEO Jia Min', 'Aya OHORI', 'GAO Fang Jie', 'SIM Yu Jin', 'ZHANG Yi Man', 'SUNG Shuo Yun', 'LIN Chun Yi', 'Toma Junior POPOV']

# 從文件中讀取 list
with open('target_players_ids.pkl', 'rb') as f:
    target_players = pickle.load(f)



player_max = 0
player_order = 0
rally_last = 0

player_data_path = "data_tables/all_dataset.csv"
player_data = pd.read_csv(player_data_path)
match_id_record = []

for index, row in player_data.iterrows():
    nan_flag = np.isnan(player_data.loc[index,'server'])
    if player_data.loc[index, 'server'] == 3 or nan_flag == True:
        for col in state_col:
            player_state_list.append(player_data.loc[index, col])
        for col in action_col:
            player_action_list.append(player_data.loc[index, col])
        # 合併為 state action pair
        player_state_action_pair = player_state_list + player_action_list
        # 新增到該 trajectory 中
        player_trajectory_data.append(player_state_action_pair.copy())
        # 準備下次的 player_state_action_pair / player_state_list / player_action_list
        player_state_action_pair = []
        player_state_list = []
        player_action_list = []
        """"""
        # 資料做完，append 進該 player 的 trajectories 中
        player_trajectories_data.append(player_trajectory_data.copy())
        player_trajectory_data = []
    else:
        for col in state_col:
            player_state_list.append(player_data.loc[index, col])
        for col in action_col:
            player_action_list.append(player_data.loc[index, col])
        # 合併為 state action pair
        player_state_action_pair = player_state_list + player_action_list
        # 新增到該 trajectory 中
        player_trajectory_data.append(player_state_action_pair.copy())
        # 準備下次的 player_state_action_pair / player_state_list / player_action_list
        player_state_action_pair = []
        player_state_list = []
        player_action_list = []

# 修正英文名字 to 數字代號
for adding_i in range(len(player_trajectories_data)):
    for adding_j in range(len(player_trajectories_data[adding_i])):
        player_number = target_players.index(player_trajectories_data[adding_i][adding_j][17])
        player_trajectories_data[adding_i][adding_j][17] = player_number

all_player_trajectories_data[player_order] = player_trajectories_data
output = open(output_name, 'wb')
pickle.dump(all_player_trajectories_data, output)
output.close()

