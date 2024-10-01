import pandas as pd
from Utils.coord2area import coord2area

def translate_coord(rally_df, Real):
    #rally_df = rally_df.reset_index()
    columns = ['rally', 'ball_round', 'player', 'type', 'player_location', 'ball_location', 'scores', 'getpoint_player']
    records = []

    for i, row in rally_df.iterrows():
        scores = [row['roundscore_A'], row['roundscore_B']]
        
        # 創建基本的記錄
        new_record = {
            'rally': row['rally'],
            'ball_round': row['ball_round'],
            'player': row['player'],
            'type': row['type'],
            'player_location': coord2area(row['player_location_x'], row['player_location_y'], Real),
            'ball_location': coord2area(row['landing_x'], row['landing_y'], Real),
            'scores': scores,
            'getpoint_player': row['getpoint_player']
        }
        records.append(new_record)

    relative_way_df = pd.DataFrame(records, columns=columns)
    relative_way_df.reset_index(drop=True, inplace=True)

    return relative_way_df

# 定義座標轉換函數
def transform_coordinates(row, player_name):
    if row['player'] == player_name:
        '''
        row['player_location_x'] = row['player_location_x'] * 177.5
        row['player_location_y'] = (row['player_location_y'] * 240) - 240
        row['landing_x'] = -(row['landing_x'] * 177.5)
        row['landing_y'] = -(row['landing_y'] * 240 - 240)
        '''
        row['scaled_landing_x'] = row['landing_x'] * 177.5 + 177.5
        row['scaled_landing_y'] = 240 - row['landing_y'] * 240 + 480
        row['scaled_moving_x'] = 177.5 - row['moving_x'] * 177.5
        row['scaled_moving_y'] = row['moving_y'] * 240 + 240

    return row
