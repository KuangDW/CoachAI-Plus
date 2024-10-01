import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def ReadFile(file):
    match = pd.read_csv(file)
    df = DataPrepocessing(match)
    df_dict = count_rally_lengths(df['rally'].explode().tolist())
    df_List = segment_dataframe(df)
    df_len = sum(df_dict.values())
    
    return df, df_dict, df_List, df_len

def ReadDataFrame(match):
    df = DataPrepocessing(match)
    df_dict = count_rally_lengths(df['rally'].explode().tolist())
    df_List = segment_dataframe(df)
    df_len = sum(df_dict.values())
    
    return df, df_dict, df_List, df_len


def DataPrepocessing(df):
    type = {'發短球': 'Serve short', '長球': 'Clear', '推撲球': 'Push Shot', 
            '殺球': 'Smash', '接殺防守': 'Smash Defence', '平球': 'Drive', 
            '接不到': 'Missed shot', '網前球': 'Net Shot', '挑球': 'Lob', 
            '切球': 'Drop', '發長球': 'Serve long'}
    
    lose_reason = {'出界': 'Out of bound', '未過網': 'Net', '對手落地致勝': 'Miss', '掛網': 'Net', '落點判斷失誤': 'Miss'}
    
    
    df['type'] = df['type'].map(type)
    df['lose_reason'] = df['lose_reason'].map(lose_reason)
    #節奏、體力消耗計算
    df['pace'] = np.nan
    df['exertion'] = np.nan
    for index, _ in df.iterrows():
        if  pd.isna(df.loc[index, "lose_reason"]) and df.loc[index, "server"] != 3:
            fly_x = df.loc[index,'hit_x']+df.loc[index,'landing_x']
            fly_y = 2-(df.loc[index,'hit_y']+df.loc[index,'landing_y']) #modified mid: +1 -> 1-hit_y + 1-landing_y
            fly_distance = (fly_x**2 + fly_y**2) ** 0.5
            df.loc[index,'fly_distance'] = fly_distance
            fly_time = df.loc[index+1, "frame_num"] - df.loc[index, "frame_num"]
            if fly_time > 0:
                df.loc[index,"pace"] = fly_distance/fly_time

        if  df.loc[index, "server"] == 2:
            move_x = df.loc[index,'player_move_x']
            move_y = df.loc[index,'player_move_y']
            move_distance = (move_x**2 + move_y**2) ** 0.5
            move_time = df.loc[index+1, "frame_num"] - df.loc[index, "frame_num"]
            if move_time > 0:
                if move_distance > 0:
                    exertion = df.loc[index-1,"pace"]/(move_distance/move_time)
                    df.loc[index,"exertion"] = exertion
        
    # Normalize pace and exertion
    scaler = MinMaxScaler()
    df[['pace', 'exertion']] = scaler.fit_transform(df[['pace', 'exertion']])
    df['exertion'] = df['exertion'].fillna(0)

    # Aggregate exertion per rally and player
    exertion_per_rally = df.groupby(['rally_id', 'player'])['exertion'].sum().reset_index()
    exertion_per_rally.rename(columns={'exertion': 'exertion_per_rally'}, inplace=True)

    # Merge exertion_per_rally back into the original dataset
    df = df.merge(exertion_per_rally, on=['rally_id', 'player'], how='left')
    df_way = df[['match_id', 'set', 'rally', 'ball_round' ,'player', 'type', 'hit_height',
            'player_location_x', 'player_location_y',
            'opponent_location_x', 'opponent_location_y', 
            'landing_x', 'landing_y', 'lose_reason', 'getpoint_player',
            'moving_x', 'moving_y', 'pace', 'exertion_per_rally']].copy()
    
    
    df_way = df_way[df_way['type'] != 'Missed shot'].reset_index(drop=True)

    #print(df_way)
    return df_way


def count_rally_lengths(rallies):
    """ Count the lengths of consecutive rally numbers and return a dictionary with the length as the key and count as the value. """
    lengths_dict = {}
    current_rally = rallies[0]
    length = 0

    for rally in rallies:
        if rally == current_rally:
            length += 1
        else:
            if length in lengths_dict:
                lengths_dict[length] += 1
            else:
                lengths_dict[length] = 1
            current_rally = rally
            length = 1

    # Add the last sequence
    if length in lengths_dict:
        lengths_dict[length] += 1
    else:
        lengths_dict[length] = 1

    return lengths_dict

def segment_dataframe(df):
    """ Segment the DataFrame by rally and reset the index for each segment, creating a list of DataFrames. """
    # Find the start indices of new rallies
    changes = df['rally'].diff().fillna(0) != 0
    starts = df[changes].index.tolist()

    if 0 not in starts:
        starts.insert(0, 0)
    ends = starts[1:] + [len(df)]

    segmented_dfs = [df.iloc[start:end].reset_index(drop=True) for start, end in zip(starts, ends)]
    return segmented_dfs
