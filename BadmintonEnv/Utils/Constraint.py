import pandas as pd
import random

class BadmintonConstraint:
    def __init__(self, file_path = './input_data/All_dataset.csv'):
        self.type_mapping = {
            'Serve short': 1, 'Clear': 2, 'Push Shot': 3, 'Smash': 4,
            'Smash Defence': 5, 'Drive': 6, 'Net Shot': 7, 'Lob': 8,
            'Drop': 9, 'Serve long': 10, 'Missed shot': 11
        }
        self.data_type_mapping = {
            '發短球': 'Serve short', '長球': 'Clear', '推撲球': 'Push Shot',
            '殺球': 'Smash', '接殺防守': 'Smash Defence', '平球': 'Drive',
            '接不到': 'Missed shot', '網前球': 'Net Shot', '挑球': 'Lob',
            '切球': 'Drop', '發長球': 'Serve long', '點扣': 'Wrist smash',
            '擋小球': 'Return Net', '放小球': 'Net shot'
        }
        
        self.data = self.read_file(file_path)
        self.action_distribution_1, self.action_distribution_2, self.action_distribution_3 = self.calculate_action_distribution(self.data)
        self.block_area = self.coord2area()


    def read_file(self, file_path):
        match_data = pd.read_csv(file_path)
        match_data = self.data_preprocessing(match_data)
        match_data['type'] = match_data['type'].map(self.type_mapping)
        return match_data


    def data_preprocessing(self, df):
        df['type'] = df['type'].map(self.data_type_mapping)
        df= df[df['type'] != 'Missed shot'].reset_index(drop=True)
        return df

    def coordinate_to_block(self, block, type):
        (x, y) = block
        if type == 'opp':
            x = -1 * x
            y = -1 * y
        
        x += 177.5
        y += 480

        hit_area = 33 # Indicates that the shot did not reach the opponent, no action
        for check_area in self.block_area:
            if x >= check_area[0][0] and y >= check_area[0][1] and x <= check_area[1][0] and y <= check_area[1][1]:
                hit_area = check_area[2]
        return hit_area

    
    def block_to_coordinate(self, block):
        for area in self.block_area:
            if area[2] == block:
                left_x, bottom_y, right_x, top_y = area[0][0], area[0][1], area[1][0], area[1][1]
                break

        x = random.uniform(left_x, right_x)
        y = random.uniform(bottom_y, top_y)

        x = (x - 177.5) * -1
        y = (y - 480) * -1

        return (x, y)
    
    def get_neighbors(self, zone):
        if zone == 1:
            return [2, 5, 6]
        elif zone == 4:
            return [3, 7, 8]
        elif zone == 21:
            return [17, 18, 22]
        elif zone == 24:
            return [19, 20, 23]
        elif zone == 2:
            return [1, 3, 5, 6, 7]
        elif zone == 3:
            return [2, 4, 6, 7, 8]
        elif zone == 22:
            return [17, 18, 19, 21, 23]
        elif zone == 3:
            return [18, 19, 20, 22, 24]
        else:
            if (zone - 1) % 4 == 0:
                return [zone-4, zone-3, zone+1, zone+4, zone+5]
            elif (zone + 1) % 4 == 1:
                return [zone-4, zone-5, zone-1, zone+3, zone+4]
            else:
                return [zone-5, zone-4, zone-3, zone-1, zone+1, zone+3, zone+4, zone+5]
    
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


    def calculate_action_distribution(self, df):
        action_distribution_1 = {}
        action_distribution_2 = {}
        action_distribution_3 = {}

        # 根據 state[0] 和 state[3] 計算 action[0] 的條件分佈 -> 球種 + 羽球位置 => 球種決定（球種上限）
        # Calculate the conditional distribution of action[0] based on state[0] and state[3] -> Shot type + Shuttlecock position => Shot type determined (shot type upper limit).
        for state_0 in df['opponent_type'].unique():
            action_distribution_1[state_0] = {}
            for state_3 in df['hit_area'].unique():
                actions_0 = df[(df['opponent_type'] == state_0) & (df['hit_area'] == state_3)]['type'].value_counts(normalize=True).reindex(range(1, 11), fill_value=0)
                action_distribution_1[state_0][state_3] = actions_0

        # 根據 action[0] 和 state[1] 計算 action[2] 的條件分佈 -> 球種決定 + 球員位置 => 落點決定（落點上限）
        # Calculate the conditional distribution of action[2] based on action[0] and state[1] -> Shot type determined + Player position => Landing point determined (landing point upper limit)
        for action_0 in df['type'].unique():
            action_distribution_2[action_0] = {}
            for state_1 in df['player_location_area'].unique():
                actions_2 = df[(df['type'] == action_0) & (df['player_location_area'] == state_1)]['landing_area'].value_counts(normalize=True).reindex(range(1, 33), fill_value=0)
                action_distribution_2[action_0][state_1] = actions_2
        
        # 根據 state[1] 計算 action[1] 的條件分佈 -> 球員位置 = 接球位置（移動上限）
        # Calculate the conditional distribution of action[1] based on state[1] -> Player position = Receiving position (movement upper limit).
        for state_1 in df['player_location_area'].unique():
            actions_1 = df[df['player_location_area'] == state_1]['hit_area'].value_counts(normalize=True).reindex(range(1, 33), fill_value=0)
            action_distribution_3[state_1] = actions_1


        return action_distribution_1, action_distribution_2, action_distribution_3


    def validate_action(self, state, action):
        state_0 = state[0]
        state_1 = self.coordinate_to_block(state[1], 'self')
        state_3 = self.coordinate_to_block(state[3], 'self')
        action_0 = action[0]
        action_1 = self.coordinate_to_block(action[1], 'self')
        action_2 = self.coordinate_to_block(action[2], 'opp')
        action_prob = action[4]  # action_prob
        reward = 0

        """ 1. P(action[0] | state[0], state[3]) """
        if state[3] != (0, 0):
            if self.action_distribution_1.get(state_0, {}).get(state_3, {}).get(action_0, 0) == 0:
                actions_0_series = self.action_distribution_1[state_0][state_3]
                if actions_0_series.sum() == 0:
                    # If all values are 0, randomly select an action_0 from 2 to 9
                    correct_action_0 = random.choice(range(2, 10))
                else:
                    # If not all are 0, find the action_0 corresponding to the maximum value
                    correct_action_0 = actions_0_series.idxmax()

                # new action_prob
                new_prob_0 = self.action_distribution_1[state_0][state_3] * action_prob
                action_prob = new_prob_0 / new_prob_0.sum()
                action = (correct_action_0, action[1], action[2], action[3], action_prob)
                reward -= 1


        """ 2. P(action[1] | state[1]) """
        if self.action_distribution_3.get(state_1, {}).get(action_1, 0) == 0:
            # If it's out of range, it's fine, just give a penalty
            reward -= 1

        """ 3. P(action[2] | action[0], state[1]) """
        historical_actions = self.action_distribution_2.get(action_0, {}).get(state_1, pd.Series())
        if historical_actions.get(action_2, 0) == 0:
            if not historical_actions.empty:
                # Find the correct action[2] from the distribution
                correct_action_2 = historical_actions.idxmax()
                action_2 = random.choice(self.get_neighbors(correct_action_2))
                landing_coordinate = self.block_to_coordinate(action_2)
            else:
                landing_coordinate = self.block_to_coordinate(action_2)

            action = (action[0], action[1], landing_coordinate, action[3], action[4])
            reward -= 1

        if reward != 0:
            return action, reward, True  # Return the corrected action and set reward = -1
        else:
            return action, reward, False  # All reasonable, return reward = 0
