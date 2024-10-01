import numpy as np
import pandas as pd
import random

class BaseServe():
    def __init__(self, fileReal = './input_data/All_dataset.csv'):
        self.serve_sa = self.agentDict(fileReal)
        self.block_area = self.coord2area()

    # data procession
    def read_file(self, fileReal):
        match_Real = pd.read_csv(fileReal)
        match_Real = self.DataPrepocessing(match_Real)
        return match_Real

    def DataPrepocessing(self, df):
        type = {'發短球': 'Serve short', '長球': 'Clear', '推撲球': 'Push Shot', 
                '殺球': 'Smash', '接殺防守': 'Smash Defence', '平球': 'Drive', 
                '接不到': 'Missed shot', '網前球': 'Net Shot', '挑球': 'Lob', 
                '切球': 'Drop', '發長球': 'Serve long', '點扣': 'Wrist smash',
                '擋小球': 'Return Net', '放小球': 'Net shot'}
        
        df['type'] = df['type'].map(type)
        df_way = df[['rally', 'ball_round' ,'player', 'type', 'opponent_type',
                    'player_location_area', 'hit_area', 'opponent_location_area', 'player_move_area', 'landing_area']].copy()    
        df_way = df_way[df_way['type'] != 'Missed shot'].reset_index(drop=True)

        return df_way

    def coord2area(self):
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

        return check_area_list


    def judge_state(self, df):
        serve_state_action_pairs = []

        df['group_id'] = (df['rally'] != df['rally'].shift(1)).cumsum()
        grouped = df.groupby('group_id')

        for _, rally in grouped:
            for i in range(len(rally)):
                if i == 0 :
                    state = (
                        0,
                        int(rally.iloc[i]['player_location_area']),
                        int(rally.iloc[i]['opponent_location_area']),
                        (0, 0)
                    )
                
                    serve_state_action_pairs.append(state)

        return serve_state_action_pairs


    def agentDict(self, fileReal):
        type_mapping = {'Serve short': 1, 'Clear': 2, 'Push Shot': 3, 'Smash':4, 'Smash Defence':5, 'Drive':6, 'Net Shot':7, 'Lob':8, 'Drop':9, 'Serve long':10, 'Missed shot':11}

        df_Real = self.read_file(fileReal)
        df_Real['type'] = df_Real['type'].map(type_mapping)
        
        serve_sa =  self.judge_state(df_Real)
        return serve_sa
    

    def _convert_block_to_coordinates(self, block, way):
        if block == 33:
            return None
        for area in self.block_area:
            if area[2] == block:
                left_x, bottom_y, right_x, top_y = area[0][0], area[0][1], area[1][0], area[1][1]
                break

        x = random.uniform(left_x, right_x)
        y = random.uniform(bottom_y, top_y)

        x = x - 177.5
        y = y - 480

        if way == 1: 
            x = x * -1
            y = y * -1 

        return (x, y)  # 反規模化回原始座標

    def serve_state(self):
        state = self.serve_sa[np.random.choice(len(self.serve_sa))]
        # 1: 對方的場，2：自己的場
        s1 = self._convert_block_to_coordinates(state[1], 2)
        s2 = self._convert_block_to_coordinates(state[2], 1)
        return (state[0], s1, s2, state[3])