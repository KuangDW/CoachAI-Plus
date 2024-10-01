import pandas as pd
import numpy as np

def ReadFile(file):
    df = pd.read_csv(file)
    df = pd.DataFrame(df)
    player_list = df['player'].unique()
    type = {'發短球': 1, '長球': 2, '推撲球': 3, 
                '殺球': 4, '接殺防守': 5, '平球': 6, 
                '接不到': 11, '網前球': 7, '挑球': 8, 
                '切球': 9, '發長球': 10}

    lose_reason = {'出界': 'Out of bound', '未過網': 'Net', '對手落地致勝': 'Miss', '掛網': 'Net', '落點判斷失誤': 'Miss', '犯規': 'foul', '對手落地判斷失誤': 'Miss'}
    win_reason = {'對手出界': 'Out of bound', '對手未過網': 'Net', '落地致勝': 'Miss', '對手掛網': 'Net', '落點判斷失誤': 'Miss', '對手落點判斷失誤': 'Miss', '對手犯規': 'foul'}
    
    df['type'] = df['type'].map(type)
    df['lose_reason'] = df['lose_reason'].map(lose_reason)
    df['win_reason'] = df['win_reason'].map(win_reason)
    df = df[df['type'] != 11].reset_index(drop=True)

    return df, player_list

def ReadDataframe(df):
    df = pd.DataFrame(df)
    player_list = df['player'].unique()
    type = {'發短球': 1, '長球': 2, '推撲球': 3, 
                '殺球': 4, '接殺防守': 5, '平球': 6, 
                '接不到': 11, '網前球': 7, '挑球': 8, 
                '切球': 9, '發長球': 10}

    lose_reason = {'出界': 'Out of bound', '未過網': 'Net', '對手落地致勝': 'Miss', '掛網': 'Net', '落點判斷失誤': 'Miss', '犯規': 'foul', '對手落地判斷失誤': 'Miss'}
    win_reason = {'對手出界': 'Out of bound', '對手未過網': 'Net', '落地致勝': 'Miss', '對手掛網': 'Net', '落點判斷失誤': 'Miss', '對手落點判斷失誤': 'Miss', '對手犯規': 'foul'}
    
    df['type'] = df['type'].map(type)
    df['lose_reason'] = df['lose_reason'].map(lose_reason)
    df['win_reason'] = df['win_reason'].map(win_reason)
    df = df[df['type'] != 11].reset_index(drop=True)

    return df, player_list

# 用一場比賽的回合來切分 dataframe
def Split_dataframe(df, rally_col='rally'):
    df = df[['rally', 'player', 'type', 'hit_area', 'landing_area', 'landing_x', 'landing_y', 'player_location_area', 'player_location_x', 'player_location_y',
             'opponent_type', 'opponent_location_area', 'opponent_location_x', 'opponent_location_y', 'hit_x', 'hit_y', 'player_move_x', 'player_move_y',
             'lose_reason', 'win_reason']]
    df_list = []
    start_idx = 0
    
    # 從第二行開始掃描每個 row，尋找新的 `rally = 1` 開始
    for i in range(1, len(df)):
        if df[rally_col].iloc[i] == 1 and df[rally_col].iloc[i - 1] != 1:
            df_list.append(df[start_idx:i])
            start_idx = i
    
    df_list.append(df[start_idx:])
    
    return df_list

def Segment_dataframe(df):
    """ Segment the DataFrame by rally and reset the index for each segment, creating a list of DataFrames. """
    # Find the start indices of new rallies
    changes = df['rally'].diff().fillna(0) != 0
    starts = df[changes].index.tolist()

    if 0 not in starts:
        starts.insert(0, 0)
    ends = starts[1:] + [len(df)]

    segmented_dfs = [df.iloc[start:end].reset_index(drop=True) for start, end in zip(starts, ends)]
    return segmented_dfs