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
import time

#%%

# df.loc[row_indexer, "col"] = values, df["col"][row_indexer] = value

def list_subtraction(p1,p2):
    point1 = p1.copy()
    point2 = p2.copy()
    v = list(map(lambda x: x[0]-x[1], zip(point1, point2)))
    return v[0], v[1]

train_or_test = "train" # str(sys.argv[1])
fold_number = 0 # int(sys.argv[2])
if train_or_test == "raw_data" and fold_number == -1:
    data_path = 'data_tables/RL_raw_data.csv'
else:
    data_path = 'data_tables/' + 'RRD_' + train_or_test + '_' + str(fold_number) + '.csv'

data = pd.read_csv(data_path)

rally_id_list = np.array(data.rally_id).tolist()
rally_id_set = list(set(rally_id_list))

for rally_id in rally_id_set:
    if data.loc[data['rally_id'] == rally_id].iloc[-1,7] != 3 and np.isnan(data.loc[data['rally_id'] == rally_id].iloc[-1,7]) == False:
        frame = (data.loc[data['rally_id'] == rally_id].iloc[-1,3]).copy()
        # data['hit_x'].loc[(data['rally_id'] == rally_id) & (data['frame_num'] == frame)] = (data['player_location_x'].loc[(data['rally_id'] == rally_id) & (data['frame_num'] == frame)]).copy()
        # data['hit_y'].loc[(data['rally_id'] == rally_id) & (data['frame_num'] == frame)] = (data['player_location_y'].loc[(data['rally_id'] == rally_id) & (data['frame_num'] == frame)]).copy()
        data.loc[(data['rally_id'] == rally_id) & (data['frame_num'] == frame), 'hit_x'] = data.loc[(data['rally_id'] == rally_id) & (data['frame_num'] == frame), 'player_location_x']
        data.loc[(data['rally_id'] == rally_id) & (data['frame_num'] == frame), 'hit_y'] = data.loc[(data['rally_id'] == rally_id) & (data['frame_num'] == frame), 'player_location_y']

print("1")

'''
座標會投射到同一個場地上 (0,0) (355,480)的範圍中 
需要調整的欄位有
1. hit_x / hit_y
2. landing_x / landing_y
3. player_location_x / player_location_y
4. opponent_location_x / opponent_location_y
把比分調整為當下比分，而非結束比分 (由下往上做調整)
作法： 如果遇到有人得分，這位選手就是這一個 rally 的 winner
會先用 dataframe_temp 存起來 winner (A or B)
who_win 是會把 dataframe_temp 的存放的結果取出
temp 是會把該 rally 的 winner 的 roundscore 取出
因為這一球是 winner 贏球，原本的 roundscore 是贏球的結果
所以會把 winner 的 roundscore (temp) - 1 去更新 
'''



# 新增欄位
col_name=data.columns.tolist()
col_name.insert(26,'player_move_x') # 選手打完球後移動的 x 向量
col_name.insert(27,'player_move_y') # 選手打完球後移動的 y 向量
col_name.insert(31,'opponent_move_x') # 對手上一顆打完球後移動的 x 向量
col_name.insert(32,'opponent_move_y') # 對手上一顆打完球後移動的 y 向量
col_name.insert(33,'ball_distance') # 球的落點和自己準備位置之間的距離
col_name.insert(34,'player_type') # 選手所使用之球種
col_name.insert(35,'opponent_type') # 對手所使用之球種 (來球球種) 
col_name.insert(36,'player_move_area') # 選手打完球後移動到的區域
col_name.insert(37,'moving_x') # 選手打完球後移動的 x 座標
col_name.insert(38,'moving_y') # 選手打完球後移動的 y 座標
col_name.insert(39,'landing_court_number') # 選手打完球後移動的 y 座標
col_name.insert(40,'ball_distance_x') # 球的落點和自己準備位置之間的x距離
col_name.insert(41,'ball_distance_y') # 球的落點和自己準備位置之間的y距離
# 添加新欄位
data=data.reindex(columns=col_name) 
print("2")

'''
因為發球時，不會接球，所以不會有hit xy，但我我認為hit xy應該可以認定是球員擊到球的位置
因此在發球時，我將hit xy 先設定為 player location xy
'''

for index, row in data.iterrows():
    # server hit xy 調整
    if data['server'][index] == 1:
        data.loc[index, 'hit_x'] = data.loc[index, 'player_location_x']
        data.loc[index, 'hit_y'] = data.loc[index, 'player_location_y']
        data.loc[index, 'hit_area'] = data.loc[index, 'player_location_area']


'''
目前設定上 hit xy = 球員位置，player_locaiton xy = 球員準備位置 (我目前假定擊球位置約等於球員位置)
計算 player move (打完這顆之後的跑位) 所以將 next row 的 opponent location - 目前的 hit xy 得到
另外因為 player move 是打完後的跑位 所以一個rally 的最後一顆是不會有的資料 (我不知道會跑到哪去)

這邊也有調整 landing height 的部份 因為有大量缺失，處理方式為 設定 next player 的 hit height (我認為可以是因為通常差不多) 

player_move, landing height 這些都是在非死球 server != 3 才去計算的
'''

for index, row in data.iterrows():  
    nan_flag = np.isnan(data.loc[index,'server'])
    if data.loc[index,'server'] != 3 and nan_flag == False:
        # player_move 計算
        player_now_location = [data.loc[index,'hit_x'], data.loc[index, 'hit_y']]
        player_next_location = [data.loc[index+1,'opponent_location_x'], data.loc[index+1, 'opponent_location_y']]
        player_move_x, player_move_y = list_subtraction(player_next_location, player_now_location)
        data.loc[index, 'player_move_x'] = player_move_x
        data.loc[index, 'player_move_y'] = player_move_y
        data.loc[index, 'player_move_area'] = data.loc[index+1,'opponent_location_area']
        data.loc[index, 'moving_x'] = data.loc[index+1,'opponent_location_x']
        data.loc[index, 'moving_y'] = data.loc[index+1,'opponent_location_y']

        # height feature 計算
        if data.loc[index, 'landing_height'] != 1 and data.loc[index, 'landing_height']!= 2:
            data.loc[index, 'landing_height'] = data.loc[index+1, 'hit_height']

print("3")

'''
同上 目前設定上 hit xy = 球員位置，player_locaiton xy = 球員準備位置 (我目前假定擊球位置約等於球員位置)
因此會需要調整player_locaiton 的資料，變成 last row 的 opponent_location (對手打球時，球員位置就是準備位置，球打過來球員才會去落地點擊球)

opponent move (對手回過來球之後，是怎麼跑的) 則是將 last player 的 hit xy - 目前的 opponent location 得到
opponent move 在一個rally的第一顆是不會有的 因為他是接發球的人 還不會有從其他地方跑位

計算 ball distance (羽球落點位置與對手回球時自己的站位的距離)，計算方式為
ball distance  = hit location - last opponent location
發球的時候 ball distance = 0 所以server = 1 的時候 不需要計算 (等等用補的)
'''

for index, row in data.iterrows():  
    nan_flag = np.isnan(data.loc[index,'server'])
    if data['server'][index] != 1 and nan_flag == False:

        data.loc[index, 'player_location_x'] = data.loc[index-1, 'opponent_location_x']
        data.loc[index, 'player_location_y'] = data.loc[index-1, 'opponent_location_y']
        data.loc[index, 'player_location_area'] = data.loc[index-1, 'opponent_location_area']

        opponent_last_location = [data.loc[index-1, 'hit_x'], data.loc[index-1, 'hit_y']]
        opponent_now_location = [data.loc[index, 'opponent_location_x'], data.loc[index, 'opponent_location_y']]
        opponent_move_x, opponent_move_y = list_subtraction(opponent_now_location, opponent_last_location)
        data.loc[index, 'opponent_move_x'] = opponent_move_x
        data.loc[index, 'opponent_move_y'] = opponent_move_y
        
        player_wait_location = [data.loc[index-1, 'opponent_location_x'], data.loc[index-1, 'opponent_location_y']]
        ball_location = [data.loc[index, 'hit_x'], data.loc[index, 'hit_y']]
        t1, t2 = list_subtraction(ball_location,player_wait_location)
        ball_distance = (t1**2 + t2**2)**0.5
        data.loc[index, 'ball_distance_x'] = t1
        data.loc[index, 'ball_distance_y'] = t2
        data.loc[index, 'ball_distance'] = ball_distance

print("4")
'''
死球後就不會移動了 
'''
for index, row in data.iterrows():  
    nan_flag = np.isnan(data.loc[index,'server'])
    if data.loc[index,'server'] == 3 or nan_flag == True:
        data.loc[index, 'player_move_area'] = data.loc[index, 'hit_area']
        data.loc[index, 'moving_x'] = data.loc[index, 'hit_x']
        data.loc[index, 'moving_y'] = data.loc[index, 'hit_y']

'''
move / ball distance add 0 
ball distance : server = 1  不會有跟球
opponent move : server = 1  對手不會移動
'''

data['ball_distance'] = data['ball_distance'].fillna(0)
data['ball_distance_x'] = data['ball_distance_x'].fillna(0)
data['ball_distance_y'] = data['ball_distance_y'].fillna(0)
data['player_move_x'] = data['player_move_x'].fillna(0)
data['player_move_y'] = data['player_move_y'].fillna(0)
data['opponent_move_x'] = data['opponent_move_x'].fillna(0)
data['opponent_move_y'] = data['opponent_move_y'].fillna(0)
data['moving_x'] = data['moving_x'].fillna(0)
data['moving_y'] = data['moving_y'].fillna(0)

print("6")

print(data.columns)
# 1/0


rally_id_list = np.array(data.rally_id).tolist()
rally_id_set = list(set(rally_id_list))
skip_flag = False
# for rally_id in rally_id_set:
#     if (data.loc[data['rally_id'] == rally_id].iloc[-1,7] != 3): # 要嘛 nan 要嘛 2
#         try:
#             frame = data.loc[data['rally_id'] == rally_id].iloc[-2,3]
#             data['server'].loc[(data['rally_id'] == rally_id) & (data['frame_num'] == frame)] = 2
            
#             filtered_df = data[data['rally_id'] == rally_id]
#             last_index = filtered_df.index[-1]
#             data.loc[last_index, 'server'] = 3
#         except:
#             print(rally_id)
#             index_to_drop = data[data['rally_id'] == rally_id].index
#             data = data.drop(index_to_drop)
#             skip_flag = True

#     if skip_flag == False:
#         if np.isnan(data.loc[data['rally_id'] == rally_id].iloc[-1,-2]):
#             match_id = data.loc[data['rally_id'] == rally_id].iloc[0,-2]
#             data['match_id'].loc[(data['rally_id'] == rally_id)] = match_id

#         if type(data.loc[data['rally_id'] == rally_id].iloc[-1,21]) == str:
#             winner = data.loc[data['rally_id'] == rally_id].iloc[-1,21]
#             data['getpoint_player'].loc[(data['rally_id'] == rally_id)] = winner
#         elif type(data.loc[data['rally_id'] == rally_id].iloc[-2,21]) == str:
#             winner = data.loc[data['rally_id'] == rally_id].iloc[-2,21]
#             data['getpoint_player'].loc[(data['rally_id'] == rally_id)] = winner
#     else:
#         skip_flag = False


for rally_id in rally_id_set:
    if (data.loc[data['rally_id'] == rally_id].iloc[-1, 7] != 3):  # 要嘛 nan 要嘛 2
        try:
            frame = data.loc[data['rally_id'] == rally_id].iloc[-2, 3]
            data.loc[(data['rally_id'] == rally_id) & (data['frame_num'] == frame), 'server'] = 2

            filtered_df = data[data['rally_id'] == rally_id]
            last_index = filtered_df.index[-1]
            data.loc[last_index, 'server'] = 3
        except:
            print(rally_id)
            index_to_drop = data[data['rally_id'] == rally_id].index
            data = data.drop(index_to_drop)
            skip_flag = True

    if not skip_flag:
        if np.isnan(data.loc[data['rally_id'] == rally_id].iloc[-1, -2]):
            match_id = data.loc[data['rally_id'] == rally_id].iloc[0, -2]
            data.loc[data['rally_id'] == rally_id, 'match_id'] = match_id

        if isinstance(data.loc[data['rally_id'] == rally_id].iloc[-1, 21], str):
            winner = data.loc[data['rally_id'] == rally_id].iloc[-1, 21]
            data.loc[data['rally_id'] == rally_id, 'getpoint_player'] = winner
        elif isinstance(data.loc[data['rally_id'] == rally_id].iloc[-2, 21], str):
            winner = data.loc[data['rally_id'] == rally_id].iloc[-2, 21]
            data.loc[data['rally_id'] == rally_id, 'getpoint_player'] = winner
    else:
        skip_flag = False

check_data = data.isnull().any()
data.to_csv("data_tables/first_process_continuous.csv", encoding = 'utf-8',index = False)
print("7")
