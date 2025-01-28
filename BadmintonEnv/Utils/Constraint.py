import random

import numpy as np
import pandas as pd


class BadmintonConstraint:
    def __init__(self, file_path="./input_data/All_dataset.csv"):
        self.type_mapping = {
            "Serve short": 1,
            "Clear": 2,
            "Push Shot": 3,
            "Smash": 4,
            "Smash Defence": 5,
            "Drive": 6,
            "Net Shot": 7,
            "Lob": 8,
            "Drop": 9,
            "Serve long": 10,
            "Missed shot": 11,
        }
        self.data_type_mapping = {
            "發短球": "Serve short",
            "長球": "Clear",
            "推撲球": "Push Shot",
            "殺球": "Smash",
            "接殺防守": "Smash Defence",
            "平球": "Drive",
            "接不到": "Missed shot",
            "網前球": "Net Shot",
            "挑球": "Lob",
            "切球": "Drop",
            "發長球": "Serve long",
            "點扣": "Wrist smash",
            "擋小球": "Return Net",
            "放小球": "Net shot",
        }

        self.valid_type_to_type = {
            "Clear": ["Clear", "Drop", "Smash"],
            "Drive": [
                "Drive",
                "Drop",
                "Net Shot",
                "Push Shot",
                "Smash",
                "Smash Defence",
            ],
            "Drop": ["Drive", "Lob", "Net Shot", "Push Shot", "Smash Defence"],
            "Lob": ["Clear", "Drive", "Drop", "Smash"],
            "Net Shot": ["Lob", "Net Shot", "Push Shot"],
            "Push Shot": [
                "Clear",
                "Drive",
                "Drop",
                "Push Shot",
                "Smash",
                "Smash Defence",
            ],
            "Serve long": ["Clear", "Drop", "Smash"],
            "Serve short": ["Lob", "Net Shot", "Push Shot"],
            "Smash": ["Net Shot", "Smash Defence"],
            "Smash Defence": [
                "Clear",
                "Drive",
                "Drop",
                "Lob",
                "Net Shot",
                "Push Shot",
                "Smash",
                "Smash Defence",
            ],
        }

        self.valid_y_to_type = [
            ["Clear", "Drive", "Drop", "Smash"],
            ["Clear", "Drive", "Drop", "Smash", "Smash Defence"],
            [
                "Clear",
                "Drive",
                "Drop",
                "Net Shot",
                "Push Shot",
                "Smash",
                "Smash Defence",
            ],
            ["Drive", "Lob", "Net Shot", "Push Shot", "Smash", "Smash Defence"],
            ["Drive", "Lob", "Net Shot", "Push Shot", "Smash Defence"],
            ["Lob", "Net Shot", "Push Shot"],
        ]

        self.hit_loc_to_type = [
            ["Clear", "Drive", "Drop", "Smash"],
            ["Clear", "Drive", "Drop", "Smash"],
            ["Clear", "Drop", "Smash"],
            ["Clear", "Drive", "Drop", "Smash"],
            ["Clear", "Drive", "Drop", "Net Shot", "Smash", "Smash Defence"],
            ["Clear", "Drive", "Drop", "Smash", "Smash Defence"],
            ["Clear", "Drive", "Drop", "Smash", "Smash Defence"],
            ["Clear", "Drive", "Drop", "Smash", "Smash Defence"],
            [
                "Clear",
                "Drive",
                "Drop",
                "Net Shot",
                "Push Shot",
                "Smash",
                "Smash Defence",
            ],
            [
                "Clear",
                "Drive",
                "Drop",
                "Net Shot",
                "Push Shot",
                "Smash",
                "Smash Defence",
            ],
            [
                "Clear",
                "Drive",
                "Drop",
                "Net Shot",
                "Push Shot",
                "Smash",
                "Smash Defence",
            ],
            [
                "Clear",
                "Drive",
                "Drop",
                "Net Shot",
                "Push Shot",
                "Smash",
                "Smash Defence",
            ],
            ["Drive", "Lob", "Net Shot", "Push Shot", "Smash Defence"],
            ["Drive", "Lob", "Net Shot", "Push Shot", "Smash", "Smash Defence"],
            ["Drive", "Lob", "Net Shot", "Push Shot", "Smash Defence"],
            ["Drive", "Lob", "Net Shot", "Push Shot", "Smash Defence"],
            ["Lob", "Net Shot", "Push Shot"],
            ["Drive", "Lob", "Net Shot", "Push Shot", "Smash Defence"],
            ["Drive", "Lob", "Net Shot", "Push Shot", "Smash Defence"],
            ["Lob", "Net Shot", "Push Shot"],
            ["Lob", "Net Shot", "Push Shot"],
            ["Lob", "Net Shot", "Push Shot"],
            ["Lob", "Net Shot", "Push Shot"],
            ["Lob", "Net Shot", "Push Shot"],
        ]

        self.invalid_type_to_land = {
            "Clear": [13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24],
            "Drive": [2],
            "Drop": [1, 2, 3, 4, 5, 6, 7],
            "Lob": [13, 18, 19, 20],
            "Net Shot": [1, 2, 3, 4, 5, 6, 7, 8, 12],
            "Push Shot": [],
            "Serve long": [10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            "Serve short": [1, 2, 3, 4, 5, 6, 7, 8, 9, 12],
            "Smash": [],
            "Smash Defence": [],
        }

        self.data = self.read_file(file_path)
        (
            self.action_distribution_1,
            self.action_distribution_2,
            self.action_distribution_3,
        ) = self.calculate_action_distribution(self.data)
        self.block_area = self.coord2area()

    def read_file(self, file_path):
        match_data = pd.read_csv(file_path)
        match_data = self.data_preprocessing(match_data)
        match_data["type"] = match_data["type"].map(self.type_mapping)
        return match_data

    def data_preprocessing(self, df):
        df["type"] = df["type"].map(self.data_type_mapping)
        df = df[df["type"] != "Missed shot"].reset_index(drop=True)
        return df

    def coordinate_to_block(self, block, type):
        (x, y) = block
        if type == "opp":
            x = -1 * x
            y = -1 * y

        x += 177.5
        y += 480

        hit_area = 33  # Indicates that the shot did not reach the opponent, no action
        for check_area in self.block_area:
            if (
                x >= check_area[0][0]
                and y >= check_area[0][1]
                and x <= check_area[1][0]
                and y <= check_area[1][1]
            ):
                hit_area = check_area[2]
        return hit_area

    def block_to_coordinate(self, block):
        for area in self.block_area:
            if area[2] == block:
                left_x, bottom_y, right_x, top_y = (
                    area[0][0],
                    area[0][1],
                    area[1][0],
                    area[1][1],
                )
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
                return [zone - 4, zone - 3, zone + 1, zone + 4, zone + 5]
            elif (zone + 1) % 4 == 1:
                return [zone - 4, zone - 5, zone - 1, zone + 3, zone + 4]
            else:
                return [
                    zone - 5,
                    zone - 4,
                    zone - 3,
                    zone - 1,
                    zone + 1,
                    zone + 3,
                    zone + 4,
                    zone + 5,
                ]

    def coord2area(self):
        area1 = [[50, 150], [104, 204], 1]
        area2 = [[104, 150], [177.5, 204], 2]
        area3 = [[177.5, 150], [251, 204], 3]
        area4 = [[251, 150], [305, 204], 4]
        row1 = [area1, area2, area3, area4]

        area5 = [[50, 204], [104, 258], 5]
        area6 = [[104, 204], [177.5, 258], 6]
        area7 = [[177.5, 204], [251, 258], 7]
        area8 = [[251, 204], [305, 258], 8]
        row2 = [area5, area6, area7, area8]

        area9 = [[50, 258], [104, 312], 9]
        area10 = [[104, 258], [177.5, 312], 10]
        area11 = [[177.5, 258], [251, 312], 11]
        area12 = [[251, 258], [305, 312], 12]
        row3 = [area9, area10, area11, area12]

        area13 = [[50, 312], [104, 366], 13]
        area14 = [[104, 312], [177.5, 366], 14]
        area15 = [[177.5, 312], [251, 366], 15]
        area16 = [[251, 312], [305, 366], 16]
        row4 = [area13, area14, area15, area16]

        area17 = [[50, 366], [104, 423], 17]
        area18 = [[104, 366], [177.5, 423], 18]
        area19 = [[177.5, 366], [251, 423], 19]
        area20 = [[251, 366], [305, 423], 20]
        row5 = [area17, area18, area19, area20]

        area21 = [[50, 423], [104, 480], 21]
        area22 = [[104, 423], [177.5, 480], 22]
        area23 = [[177.5, 423], [251, 480], 23]
        area24 = [[251, 423], [305, 480], 24]
        row6 = [area21, area22, area23, area24]

        area25 = [[305, 366], [355, 480], 25]
        area26 = [[305, 204], [355, 366], 26]
        area27 = [[305, 0], [355, 204], 27]
        area28 = [[177.5, 0], [305, 150], 28]
        row7 = [area25, area26, area27, area28]

        area29 = [[0, 366], [50, 480], 29]
        area30 = [[0, 204], [50, 366], 30]
        area31 = [[0, 0], [50, 204], 31]
        area32 = [[50, 0], [177.5, 150], 32]
        row8 = [area29, area30, area31, area32]

        check_area_list = row1 + row2 + row3 + row4 + row5 + row6 + row7 + row8

        # Add calculated center coordinates for each area
        for area in check_area_list:
            x_center = (area[0][0] + area[1][0]) / 2
            y_center = (area[0][1] + area[1][1]) / 2
            area.append((x_center, y_center))  # Append center coordinates to each area

        return check_area_list

    def find_nearest_valid_area(self, invalid_land_x, invalid_land_y, valid_area_ids):

        # Calculate distances to all valid areas
        min_distance = float("inf")
        nearest_area_id = 0  # Default to itself in case no valid area is found

        for area in self.block_area:
            if area[2] in valid_area_ids:
                valid_center = area[3]
                distance = (
                    (valid_center[0] - invalid_land_x) ** 2
                    + (valid_center[1] - invalid_land_y) ** 2
                ) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_area_id = area[2]

        return nearest_area_id

    def correct_position(
        self,
        action_1_x,
        action_1_y,
        action_3_x,
        action_3_y,
        max_total=1.6,
    ):
        # Calculate the movement vector (dx, dy)
        dx = (action_3_x - action_1_x) / 177.5
        dy = (action_3_y - action_1_y) / 240

        total_distance = (dx**2 + dy**2) ** 0.5

        # If no limit is exceeded, return the original position
        if total_distance <= max_total:
            return action_3_x, action_3_y
        else:
            scale_total = max_total / total_distance
            corrected_dx = dx * scale_total
            corrected_dy = dy * scale_total

        # Calculate the corrected position
        corrected_x = action_1_x + corrected_dx * 177.5
        corrected_y = action_1_y + corrected_dy * 240

        return corrected_x, corrected_y

    def calculate_action_distribution(self, df):
        action_distribution_1 = {}
        action_distribution_2 = {}
        action_distribution_3 = {}

        # 根據 state[0] 和 state[3] - state[1] 和 state[1] 計算 action[0] 的條件分佈 -> 球種 + 跑動方向 + 擊球點 => 球種決定（球種上限）
        # Calculate the conditional distribution of action[0] based on state[0] and state[3] - state[1] -> Shot type + Player y movement => Shot type determined (shot type upper limit).
        for state_0, type_valid_types in self.valid_type_to_type.items():
            state_0 = self.type_mapping[state_0]
            # Initialize action distribution array for this state_0
            action_distribution_1[state_0] = [
                [[0] * 10 for _ in range(6)] for _ in range(24)
            ]

            # For each y-movement range and hit location, determine valid actions
            for y_move_index, y_valid_types in enumerate(self.valid_y_to_type):
                for hit_loc_index, hit_loc_valid_types in enumerate(
                    self.hit_loc_to_type
                ):
                    # Compute the intersection of valid types from both constraints
                    valid_actions = set(type_valid_types).intersection(
                        y_valid_types, hit_loc_valid_types
                    )

                    # Set corresponding indices to 1 for valid actions in action_distribution_1
                    for action in valid_actions:
                        action_index = self.type_mapping[action]
                        action_distribution_1[state_0][hit_loc_index][y_move_index][
                            # action_index
                            action_index-1
                        ] = 1

        # 根據 action[0] 和 state[3] 計算 action[2] 的條件分佈 -> 球種決定 + 擊球點 => 落點決定（落點上限）
        # Calculate the conditional distribution of action[2] based on action[0] and state[3] -> Shot type determined + Hit position => Landing point determined (landing point upper limit)
        for action_0, invalid_areas in self.invalid_type_to_land.items():
            # Start with a 24-element array of 1s (all landing areas valid)
            action_0 = self.type_mapping[action_0]
            action_distribution_2[action_0] = [1] * 24

            # Set landing areas specified in invalid_type_to_land to 0
            for invalid_area in invalid_areas:
                action_distribution_2[action_0][
                    invalid_area - 1
                ] = 0  # Convert to 0-based

        return action_distribution_1, action_distribution_2, action_distribution_3

    def validate_action(self, state, action):
        t2c = {1: 1, 2: 10, 3: 2, 4: 4, 5: 9, 6: 8,
           7: 6, 8: 7, 9: 3, 10: 5, 11: 11}
        
        action = list(action)
        action[0] = t2c.get(int(action[0]), "Unknown Category") 
        action = tuple(action)

        state_0 = state[0]
        (state_1_x, state_1_y) = state[1]
        (_, state_3_y) = state[3]
        action_0 = action[0]
        (action_1_x, action_1_y) = action[1]
        (action_2_x, action_2_y) = action[2]
        (action_3_x, action_3_y) = action[3]
        action_prob = action[4]  # action_prob
        reward = 0

        """ 1. 球種決定（球種上限）"""
        action_1 = self.coordinate_to_block(action[1], "self") - 1
        y_movement = (state_3_y - state_1_y) / 240
        if y_movement < -0.5:
            y_move_index = 0
        elif -0.5 <= y_movement < -0.25:
            y_move_index = 1
        elif -0.25 <= y_movement < 0:
            y_move_index = 2
        elif 0 <= y_movement < 0.25:
            y_move_index = 3
        elif 0.25 <= y_movement < 0.5:
            y_move_index = 4
        else:
            y_move_index = 5

        valid_mask = self.action_distribution_1.get(
            state_0, [[[0] * 10 for _ in range(6)] for _ in range(24)]
        )[action_1][y_move_index]

        if state[3] != (0, 0):
            if (
                self.action_distribution_1.get(
                    state_0, [[[0] * 10 for _ in range(6)] for _ in range(24)]
                )[action_1][y_move_index][action_0-1]
                == 0
            ):
                # print(state_0)
                # print(action_0)
                # print(y_move_index)
                valid_mask = np.array(valid_mask)
                if valid_mask.sum() == 0:
                    # If all values are 0, randomly select an action_0 from 2 to 9
                    correct_action_0 = random.choice(range(2, 10))
                    action_prob = tuple(0 for _ in range(10))
                    # print("violation 1-1")
                    reward -= 10
                else:
                    new_prob_0 = valid_mask * np.array(action_prob)
                    action_prob = new_prob_0 / new_prob_0.sum()
                    correct_action_0 = np.argmax(action_prob)
                    # print(valid_mask)
                    # print(correct_action_0)
                    # print("violation 1-2")
                action = (
                    correct_action_0,
                    action[1],
                    action[2],
                    action[3],
                    tuple(action_prob),
                )
                reward -= 1

        """ 2. 落點決定（落點上限）"""
        action_2 = self.coordinate_to_block(action[2], "opp")
        if self.action_distribution_2.get(action[0], [1] * 24)[action_2 - 1] == 0:
            valid_areas = [
                i
                for i, is_valid in enumerate(
                    self.action_distribution_2[action[0]], start=1
                )
                if is_valid == 1
            ]
            nearest_valid_area = self.find_nearest_valid_area(
                action_2_x, action_2_y, valid_areas
            )
            if nearest_valid_area == 0:
                correct_action_2 = random.choice(range(1, 24))
                reward -= 10
            else:
                correct_action_2 = self.block_to_coordinate(nearest_valid_area)
            action = (action[0], action[1], correct_action_2, action[3], action_prob)
            reward -= 1
            # print("violation 2")

        """ 3. 回位決定（移動上限）"""
        corrected_x, corrected_y = self.correct_position(
            action_1_x, action_1_y, action_3_x, action_3_y
        )
        if (corrected_x, corrected_y) != (action_3_x, action_3_y):
            action = (
                action[0],
                action[1],
                action[2],
                (corrected_x, corrected_y),
                action_prob,
            )
            reward -= 1
            # print("violation 3")

        if reward != 0:
            return (
                action,
                reward,
                True,
            )  # Return the corrected action and set reward = -1
        else:
            return action, reward, False  # All reasonable, return reward = 0
