import pandas as pd
import numpy as np

def ReadFile(file):
    df = pd.read_csv(file)
    df = pd.DataFrame(df)
    player_list = df['player'].unique()
    # print('Players:', player_list)

    type = {'發短球': 'Serve short', '長球': 'Clear', '推撲球': 'Push Shot', 
                '殺球': 'Smash', '接殺防守': 'Smash Defence', '平球': 'Drive', 
                '接不到': 'Missed shot', '網前球': 'Net Shot', '挑球': 'Lob', 
                '切球': 'Drop', '發長球': 'Serve long'}
        
    lose_reason = {'出界': 'Out of bound', '未過網': 'Net', '對手落地致勝': 'Miss', '掛網': 'Net', '落點判斷失誤': 'Miss'}
    
    df['type'] = df['type'].map(type)
    df['lose_reason'] = df['lose_reason'].map(lose_reason)
    df = df[df['type'] != 'Missed shot'].reset_index(drop=True)

    return df, player_list

def ReadDataFrame(df):
    df = pd.DataFrame(df)
    player_list = df['player'].unique()
    # print('Players:', player_list)

    type = {'發短球': 'Serve short', '長球': 'Clear', '推撲球': 'Push Shot', 
                '殺球': 'Smash', '接殺防守': 'Smash Defence', '平球': 'Drive', 
                '接不到': 'Missed shot', '網前球': 'Net Shot', '挑球': 'Lob', 
                '切球': 'Drop', '發長球': 'Serve long'}
        
    lose_reason = {'出界': 'Out of bound', '未過網': 'Net', '對手落地致勝': 'Miss', '掛網': 'Net', '落點判斷失誤': 'Miss'}
    
    df['type'] = df['type'].map(type)
    df['lose_reason'] = df['lose_reason'].map(lose_reason)
    df = df[df['type'] != 'Missed shot'].reset_index(drop=True)

    return df, player_list


def coord2area(point_x, point_y):
    mistake_landing_area = 33

    point_x = (point_x * (355/2)) + (355/2)
    point_y = (point_y * 240) + 240

    area1 = [[50,150],[104,204],1]
    area2 = [[104,150],[177.5,204],2]
    area3 = [[177.5,150],[251,204],3]
    area4 = [[251,150],[305,204],4]
    row1 = [area1, area2, area3, area4]

    area5 = [[50,204],[104,258],5]
    area6 = [[104,204],[177.5,258],6]
    area7 = [[177.5,204],[251,258],7]
    area8 = [[251,204],[305,258],8]
    row2 = [area5, area6, area7, area8]

    area9 = [[50,258],[104,312],9]
    area10 = [[104,258],[177.5,312],10]
    area11 = [[177.5,258],[251,312],11]
    area12 = [[251,258],[305,312],12]
    row3 = [area9, area10, area11, area12]
    
    area13 = [[50,312],[104, 366],13]
    area14 = [[104,312],[177.5,366],14]
    area15 = [[177.5,312],[251,366],15]
    area16 = [[251,312],[305,366],16]
    row4 = [area13, area14, area15, area16]

    area17 = [[50,366],[104,423],17]
    area18 = [[104,366],[177.5,423],18]
    area19 = [[177.5,366],[251,423],19]
    area20 = [[251,366],[305,423],20]
    row5 = [area17, area18, area19, area20]

    area21 = [[50,423],[104,480],21]
    area22 = [[104,423],[177.5,480],22]
    area23 = [[177.5,423],[251,480],23]
    area24 = [[251,423],[305,480],24]
    row6 = [area21, area22, area23, area24]

    area25 = [[305,366],[355,480],25]
    area26 = [[305,204],[355,366],26]
    area27 = [[305,0],[355,204],27]
    area28 = [[177.5,0],[305,150],28]
    row7 = [area25, area26, area27, area28]

    area29 = [[0,366],[50,480],29]
    area30 = [[0,204],[50,366],30]
    area31 = [[0,0],[50,204],31]
    area32 = [[50,0],[177.5,150],32]
    row8 = [area29, area30, area31, area32]

    check_area_list = row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8
    hit_area = mistake_landing_area
    for check_area in check_area_list:
        if point_x >= check_area[0][0] and point_y >= check_area[0][1] and point_x <= check_area[1][0] and point_y <= check_area[1][1]:
            hit_area = check_area[2]
    return hit_area


def get_other_player(given_player, player_list):
    if given_player == player_list[0]:
        return player_list[1]
    elif given_player == player_list[1]:
        return player_list[0]


def translate_coord(rally_df, player_list):
    #rally_df = rally_df.reset_index()
    columns = ['rally', 'ball_round', 'player', 'type', 'player_location', 'ball_location', 'scores', 'getpoint_player']
    records = []

    for i, row in rally_df.iterrows():
        scores = [row['player_score'], row['opponent_score']]
        # # 把 getpointer 的player改成名字
        # if i == 0:
        #     p = row['player']
        #     if row['getpoint_player'] == 'player':
        #         winner = p
        #     else:
        #         winner = get_other_player(p, player_list)

        # 創建基本的記錄
        new_record = {
            'rally': row['rally'],
            'ball_round': row['ball_round'],
            'player': row['player'],
            'type': row['type'],
            'player_location': coord2area(row['player_location_x'], row['player_location_y']),
            'ball_location': coord2area(row['landing_x'], row['landing_y']),
            'scores': scores,
            'getpoint_player': row['getpoint_player'],
        }
        records.append(new_record)

    df = pd.DataFrame(records, columns=columns)
    df.reset_index(drop=True, inplace=True)

    return df

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
