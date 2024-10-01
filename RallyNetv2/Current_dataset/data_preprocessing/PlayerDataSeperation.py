
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

data_path = 'data_tables/second_process_discrete.csv' # first_process_continuous, second_process_discrete, 
match_path = 'data/match.csv'
match_data = pd.read_csv(match_path)
data = pd.read_csv(data_path)
print(len(data))

target_players = data['player'].unique().tolist()
print(target_players)

with open('target_players_ids.pkl', 'wb') as f:
    pickle.dump(target_players, f)

# ['Kento MOMOTA', 'CHOU Tien Chen', 'Anthony Sinisuka GINTING', 'CHEN Long', 'CHEN Yufei', 'TAI Tzu Ying', 'Viktor AXELSEN', 'Anders ANTONSEN', 'PUSARLA V. Sindhu', 'WANG Tzu Wei', 'Khosit PHETPRADAB', 'Jonatan CHRISTIE', 'NG Ka Long Angus', 'SHI Yuqi', 'Ratchanok INTANON', 'An Se Young', 'Busanan ONGBAMRUNGPHAN', 'Mia BLICHFELDT', 'LEE Zii Jia', 'LEE Cheuk Yiu', 'Rasmus GEMKE', 'Michelle LI', 'Supanida KATETHONG', 'Carolina MARIN', 'Pornpawee CHOCHUWONG', 'Sameer VERMA', 'Neslihan YIGIT', 'Hans-Kristian Solberg VITTINGHUS', 'LIEW Daren', 'Evgeniya KOSETSKAYA', 'KIDAMBI Srikanth', 'Soniia CHEAH', 'Gregoria Mariska TUNJUNG', 'Akane YAMAGUCHI', 'HE Bingjiao', '胡佑齊', '張允澤', '許喆宇', '陳政佑', '林祐賢', '李佳豪', 'LOH Kean Yew', 'Lakshya SEN', 'Kunlavut VITIDSARN', 'WANG Hong Yang', 'Kodai NARAOKA', 'JEON Hyeok Jin', 'Wen Chi Hsu', 'Nozomi Okuhara', 'WANG Zhi Yi', 'PRANNOY H. S.', 'Chico Aura DWI WARDOYO', 'LU Guang Zu', 'ZHAO Jun Peng', 'Kenta NISHIMOTO', 'NG Tze Yong', 'Victor SVENDSEN', 'WENG Hong Yang', 'Aakarshi KASHYAP', 'LI Shi Feng', 'KIM Ga Eun', 'HAN Yue', 'Other', 'NYCU', 'Kanta TSUNEYAMA', 'YEO Jia Min', 'Aya OHORI', 'GAO Fang Jie', 'SIM Yu Jin', 'ZHANG Yi Man', 'SUNG Shuo Yun', 'LIN Chun Yi', 'Toma Junior POPOV']

def type2cat(shot_type):
    t2c = {'發短球': 1, '發長球': 2, '長球': 3, '殺球': 4, '切球': 5, '挑球': 6,
           '平球': 7, '網前球': 8, '推撲球': 9, '接殺防守': 10, '接不到': 11}
    return t2c[shot_type]

'''
target_player_max 是 ball_round 的最大值 (該球員在一個rally中最多打到這麼多顆)
'''
# target_players = ['CHOU Tien Chen', 'LOH Kean Yew', 'LIN Chun Yi', 'Kodai NARAOKA']
player_name= target_players[0]

# print(player_name)
# 該球員的比賽獨立出來 列出 match id (該球員比過的比賽)
match_id = set((data.loc[data['player'] == player_name])['match_id'])
match_id_record = list(match_id).copy()
# print(match_id)
# 列出 match id
player_matches_data = data.loc[data['match_id'].isin(list(match_id))]
player_matches_data = player_matches_data.sort_index().reset_index(drop=True)

# 然後把分數改為特定 player : player_score / opponent_score
player_matches_data = player_matches_data.rename(columns={'roundscore_A': 'player_score', 'roundscore_B': 'opponent_score'}, inplace=False)
adding_loss_list = []

for index, row in player_matches_data.iterrows():
    # 這排找到 match id 去 match.csv 找到對應的row
    search_id = player_matches_data.loc[index, 'match_id']
    target_row = match_data.loc[match_data['id'] == int(search_id)]
    # 找到誰是 winner(A) 誰是 loser(B)  
    # 如果發現該 winner 不是這次的 player 的話，就會知道這個人分數會在 roundscore_B
    # 我們就會把 roundscore_A 和 roundscore_B 對調
    if target_row.iloc[0]['winner'] != player_name:
        temp = player_matches_data.loc[index, 'player_score'] 
        player_matches_data.loc[index, 'player_score'] = player_matches_data.loc[index, 'opponent_score'] 
        player_matches_data.loc[index, 'opponent_score'] = temp
        
        # 把贏球的根據目前的plyaer去調整
        # 因為現在贏球是A的話 表示是對手 因此 getpoint_player = opponent
        if player_matches_data.loc[index, 'getpoint_player'] == 'A':
            player_matches_data.loc[index, 'getpoint_player'] = 'opponent'
        elif player_matches_data.loc[index, 'getpoint_player'] == 'B':
            player_matches_data.loc[index, 'getpoint_player'] = 'player'
    else:
        if player_matches_data.loc[index, 'getpoint_player'] == 'A':
            player_matches_data.loc[index, 'getpoint_player'] = 'player'
        elif player_matches_data.loc[index, 'getpoint_player'] == 'B':
            player_matches_data.loc[index, 'getpoint_player'] = 'opponent'

    player_server = player_matches_data.loc[index, 'server']
    player_ball_type = player_matches_data.loc[index, 'type']
    player_win = player_matches_data.loc[index, 'getpoint_player']

    '''
    因為死球時，有可能是贏球也可能是輸球
    會先判斷輸贏，然後搭配球種type產生新的feature: player_type
    opponent_type是對手打來的球，也就是 last row 的 player_type
    '''
    
    flag = np.isnan(player_server)
    if flag == True:
        player_server = 3
    
    if int(player_server) == 3 and player_win == 'opponent':
        # lose 自身失誤 => 最後一顆自己打的，而且贏球是對手
        pluson = 0 #20
        player_type = pluson + int(type2cat(player_ball_type))
        player_matches_data.loc[index, 'player_type'] = int(player_type)
        player_matches_data.loc[index, 'opponent_type'] = player_matches_data.loc[index-1, 'player_type']
    elif int(player_server) == 3 and player_win == 'player':
        # win 自身致勝 => 最後一顆自己打的，而且贏球是自己
        pluson = 0 #10
        player_type = pluson + int(type2cat(player_ball_type))
        player_matches_data.loc[index, 'player_type'] = int(player_type)
        player_matches_data.loc[index, 'opponent_type'] = player_matches_data.loc[index-1, 'player_type']
    elif int(player_server) != 3:
        # keep 普通來回 = 只要非死球，就是普通來回，直接紀錄球種
        pluson = 0
        player_type = pluson + int(type2cat(player_ball_type))
        player_matches_data.loc[index, 'player_type'] = int(player_type)

        if int(player_server) == 1:
            player_matches_data.loc[index, 'opponent_type'] = 0 # 接發球為 無動作
        if int(player_server) == 2:
            player_matches_data.loc[index, 'opponent_type'] = player_matches_data.loc[index-1, 'player_type']


# opponent_type : server = 1 不會接球 不會有接球
# player_matches_data['opponent_type'] = player_matches_data['opponent_type'].fillna(0)

# 分數對調之後 添加 score_status 紀錄 領先幾分 平手 落後幾分
col_name = player_matches_data.columns.tolist()  
col_name.insert(4,'score_status')
player_matches_data=player_matches_data.reindex(columns=col_name)  
for index, row in player_matches_data.iterrows():
    player_matches_data.loc[index, 'score_status'] = player_matches_data.loc[index, 'player_score'] - player_matches_data.loc[index, 'opponent_score'] 

player_final_data = player_matches_data.copy()
print(len(player_final_data))

for i in range(1, len(target_players)):
    adding_loss_list = []
    player_name= target_players[i]
    # print(player_name)
    # 該球員的比賽獨立出來 列出 match id (該球員比過的比賽)
    match_id = set((data.loc[data['player'] == player_name])['match_id'])
    match_id = match_id.difference(set(match_id_record))
    # print(match_id)
    # 列出 match id
    player_matches_data = data.loc[data['match_id'].isin(list(match_id))]
    player_matches_data = player_matches_data.sort_index().reset_index(drop=True)

    match_id_record.extend(list(match_id))

    # 然後把分數改為特定 player : player_score / opponent_score
    player_matches_data = player_matches_data.rename(columns={'roundscore_A': 'player_score', 'roundscore_B': 'opponent_score'}, inplace=False)
    for index, row in player_matches_data.iterrows():
        # 這排找到 match id 去 match.csv 找到對應的row
        search_id = player_matches_data.loc[index, 'match_id']
        target_row = match_data.loc[match_data['id'] == int(search_id)]
        # 找到誰是 winner(A) 誰是 loser(B)  
        # 如果發現該 winner 不是這次的 player 的話，就會知道這個人分數會在 roundscore_B
        # 我們就會把 roundscore_A 和 roundscore_B 對調
        if target_row.iloc[0]['winner'] != player_name:
            temp = player_matches_data.loc[index, 'player_score'] 
            player_matches_data.loc[index, 'player_score'] = player_matches_data.loc[index, 'opponent_score'] 
            player_matches_data.loc[index, 'opponent_score'] = temp
            # 把贏球的根據目前的plyaer去調整
            # 因為現在贏球是A的話 表示是對手 因此 getpoint_player = opponent
            if player_matches_data.loc[index, 'getpoint_player'] == 'A':
                player_matches_data.loc[index, 'getpoint_player'] = 'opponent'
            elif player_matches_data.loc[index, 'getpoint_player'] == 'B':
                player_matches_data.loc[index, 'getpoint_player'] = 'player'
        else:
            if player_matches_data.loc[index, 'getpoint_player'] == 'A':
                player_matches_data.loc[index, 'getpoint_player'] = 'player'
            elif player_matches_data.loc[index, 'getpoint_player'] == 'B':
                player_matches_data.loc[index, 'getpoint_player'] = 'opponent'
        
        player_server = player_matches_data.loc[index, 'server']
        player_ball_type = player_matches_data.loc[index, 'type']
        player_win = player_matches_data.loc[index, 'getpoint_player']

        '''
        因為死球時，有可能是贏球也可能是輸球
        會先判斷輸贏，然後搭配球種type產生新的feature: player_type
        opponent_type是對手打來的球，也就是 last row 的 player_type
        '''

        flag = np.isnan(player_server)
        if flag == True:
            player_server = 3
    
        if int(player_server) == 3 and player_win == 'opponent':
            # lose 自身失誤 => 最後一顆自己打的，而且贏球是對手
            pluson = 0 #20
            player_type = pluson + int(type2cat(player_ball_type))
            player_matches_data.loc[index, 'player_type'] = int(player_type)
            player_matches_data.loc[index, 'opponent_type'] = player_matches_data.loc[index-1, 'player_type']
        elif int(player_server) == 3 and player_win == 'player':
            # win 自身致勝 => 最後一顆自己打的，而且贏球是自己
            pluson = 0 #10
            player_type = pluson + int(type2cat(player_ball_type))
            player_matches_data.loc[index, 'player_type'] = int(player_type)
            player_matches_data.loc[index, 'opponent_type'] = player_matches_data.loc[index-1, 'player_type']
        elif int(player_server) != 3:
            # keep 普通來回 = 只要非死球，就是普通來回，直接紀錄球種
            pluson = 0
            player_type = pluson + int(type2cat(player_ball_type))
            player_matches_data.loc[index, 'player_type'] = int(player_type)
            if int(player_server) == 1:
                player_matches_data.loc[index, 'opponent_type'] = 0 # 接發球為 無動作
            if int(player_server) == 2:
                player_matches_data.loc[index, 'opponent_type'] = player_matches_data.loc[index-1, 'player_type']

        
    # 分數對調之後 添加 score_status 紀錄 領先幾分 平手 落後幾分
    col_name=player_matches_data.columns.tolist()  
    col_name.insert(4,'score_status')
    player_matches_data=player_matches_data.reindex(columns=col_name)  
    for index, row in player_matches_data.iterrows():
        player_matches_data.loc[index, 'score_status'] = player_matches_data.loc[index, 'player_score'] - player_matches_data.loc[index, 'opponent_score'] 
    player_final_data = pd.concat([player_final_data, player_matches_data], ignore_index = True)
    print(len(player_final_data))

player_final_data.to_csv("data_tables/all_dataset.csv", encoding = 'utf-8',index = False)
