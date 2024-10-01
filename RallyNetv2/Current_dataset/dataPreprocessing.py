import pandas as pd
import numpy as np
import math
import ast
import os
import re

from typing import Tuple, Literal, Dict

# 客觀 -> 主觀
"""
客觀座標轉為主觀座標
客觀座標: (0, 0) 在場地左上角，以下為正
主觀座標: (0, 0) 在場地中心
player 1 在上半場， 2 在下半場
"""
def obj2subj_coord(objective_coord: Tuple[float, float]):
    x, y = objective_coord
    if y < 960/2:
        x -= 177.5
        y -= 240
    elif y >= 960/2:
        # rotate 180 deg
        x = 355 - x
        y = 960 - y
        x -= 177.5
        y -= 240
    else:
        NotImplementedError
    return x, y

def drop_na_rally(df, columns=[]):
    """Drop rallies which contain na value in columns."""
    df = df.copy()
    for column in columns:
        rallies = df[df[column].isna()]['rally_id']
        df = df[~df['rally_id'].isin(rallies)]
    df = df.reset_index(drop=True)
    return df

def preprocess(filename: str) :

    matches = pd.read_csv(filename)

    ### move from DyMF ###
    matches['rally_id'] = matches.groupby(['match_id', 'set', 'rally']).ngroup()

    # Drop flaw rally
    if 'flaw' in matches.columns:
        flaw_rally = matches[matches['flaw'].notna()]['rally_id']
        matches = matches[~matches['rally_id'].isin(flaw_rally)]
        matches = matches.reset_index(drop=True)

    # Drop unknown ball type
    unknown_rally = matches[matches['type'] == '未知球種']['rally_id']
    matches = matches[~matches['rally_id'].isin(unknown_rally)]
    matches = matches.reset_index(drop=True)

    # Drop hit_area at outside
    outside_area = [10, 11, 12, 13, 14, 15, 16]
    matches.loc[matches['server'] == 1, 'hit_area'] = 7
    for area in outside_area:
        outside_rallies = matches.loc[matches['hit_area'] == area, 'rally_id']
        matches = matches[~matches['rally_id'].isin(outside_rallies)]
        matches = matches.reset_index(drop=True)
    # Deal with hit_area convert hit_area to integer
    matches = drop_na_rally(matches, columns=['hit_area'])
    matches['hit_area'] = matches['hit_area'].astype(float).astype(int)

    # Convert landing_area outside to 10 and to integer
    matches = drop_na_rally(matches, columns=['landing_area'])
    for area in outside_area:
        matches.loc[matches['landing_area'] == area, 'landing_area'] = 10
    matches['landing_area'] = matches['landing_area'].astype(float).astype(int)

    # Deal with ball type. Convert ball types to general version (10 types)
    # Convert 小平球 to 平球 because of old version
    matches['type'] = matches['type'].replace('小平球', '平球')
    combined_types = {'切球': '切球', '過度切球': '切球', '點扣': '殺球', '殺球': '殺球', '平球': '平球', '後場抽平球': '平球', '擋小球': '接殺防守',
                '防守回挑': '接殺防守', '防守回抽': '接殺防守', '放小球': '網前球', '勾球': '網前球', '推球': '推撲球', '撲球': '推撲球'}
    matches['type'] = matches['type'].replace(combined_types)

    # Fill zero value in backhand
    matches['backhand'] = matches['backhand'].fillna(value=0)
    matches['backhand'] = matches['backhand'].astype(float).astype(int)

    # Fill zero value in aroundhead
    matches['aroundhead'] = matches['aroundhead'].fillna(value=0)
    matches['aroundhead'] = matches['aroundhead'].astype(float).astype(int)

    # Convert ball round type to integer
    matches['ball_round'] = matches['ball_round'].astype(float).astype(int)

    ################
    new_rows: Dict[str, list] = {'hit_x':[], 'hit_y': [], 'landing_x':[], 'landing_y': [], 'player_location_x':[], 'player_location_y':[], 
                             'opponent_location_x':[], 'opponent_location_y':[],'type':[], 'rally' :[],
                             'rally_id' : [], 'ball_round': [],'player':[], 'set':[]}
    for i, row in matches.iterrows():        
        # 改用主觀座標系
        landing_x, landing_y = obj2subj_coord((row['landing_x'], row['landing_y']))
        hit_x, hit_y = obj2subj_coord((row['hit_x'], row['hit_y']))
        player_x, player_y = obj2subj_coord((row['player_location_x'], row['player_location_y']))
        opponent_x, opponent_y = obj2subj_coord((row['opponent_location_x'], row['opponent_location_y']))

        # normalize到 -1 ~ +1
        hit_x /= 177.5
        landing_x /= 177.5
        player_x /= 177.5
        opponent_x /= 177.5
        hit_y /= 240
        landing_y /= 240
        player_y /= 240
        opponent_y /= 240

        matches.at[i, 'hit_x'], matches.at[i,'hit_y'] = hit_x, hit_y
        matches.at[i, 'landing_x'], matches.at[i,'landing_y'] = landing_x, landing_y
        matches.at[i, 'player_location_x'], matches.at[i,'player_location_y'] = player_x, player_y
        matches.at[i, 'opponent_location_x'], matches.at[i, 'opponent_location_y'] = opponent_x, opponent_y

        # win_reason 有 ['對手未過網', '對手落點判斷失誤', '落地判斷失誤', '對手出界', '對手掛網', '對手犯規', '落地致勝'] 幾種
        # 除了 對手出界 外都沒有下一球落點，補上球種: '接不到'
        # 對手犯規 貌似不一定有無下一球，但數量很少(8球)，可略
        if pd.notna(row['win_reason']) and row['win_reason'] != '對手出界':
            new_rows['hit_x'].append(0.)
            new_rows['hit_y'].append(0.)
            new_rows['landing_x'].append(0.)
            new_rows['landing_y'].append(1.)
            new_rows['player_location_x'].append(0.)    # (0, 0) 表示打完失誤時，預設回到場地中心
            new_rows['player_location_y'].append(0.)    # (0, 0) 表示打完失誤時，預設回到場地中心
            new_rows['opponent_location_x'].append(0.)  # (0, 0) 表示打完失誤時，預設回到場地中心
            new_rows['opponent_location_y'].append(0.)  # (0, 0) 表示打完失誤時，預設回到場地中心
            new_rows['type'].append('接不到')
            new_rows['rally'].append(row['rally'])
            new_rows['rally_id'].append(row['rally_id'])
            new_rows['ball_round'].append(row['ball_round']+1)
            new_rows['player'].append(matches.at[i-1, 'player'])
            new_rows['set'].append(row['set'])
    matches = pd.concat([matches, pd.DataFrame(new_rows)])#.reset_index(drop=True)
    
    matches = matches.sort_values(by=['rally_id','rally', 'ball_round'])
    matches.to_csv('data_tables/RRD_train_0.csv', index=False)

if __name__ == '__main__':
    preprocess('data_tables/raw_dataset.csv')
