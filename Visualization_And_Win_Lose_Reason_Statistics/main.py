import pandas as pd
import numpy as np
import os
from Utils.ShotTypeDistribution import Player_ShotType_histogram
from Utils.Dataprocessing import ReadDataframe, Split_dataframe, Segment_dataframe
from Utils.StatWinLose import *


def Visualize(match, state, dest_folder = './Visualization_And_Win_Lose_Reason_Statistics/Result'):
    # data processing
    df, player_list = ReadDataframe(match)
    df_list = Split_dataframe(df)
    rally_list = Segment_dataframe(df)

    Win_df = pd.DataFrame()
    Lose_df = pd.DataFrame()

    file_dict = {}
    os.makedirs(dest_folder, exist_ok=True)

    # Plot
    # 1. Shot type distribution
    for player in player_list:
        f1_list = Player_ShotType_histogram(df, player, dest_folder)

    # 2. win/lose state action
        for d in df_list:
            win_df, loss_df = Tally_win_loss(d, player)
            Win_df = pd.concat([Win_df, win_df], ignore_index=True)
            Lose_df = pd.concat([Lose_df, loss_df], ignore_index=True)
        
        RWin = pd.DataFrame()
        RLose = pd.DataFrame()

        for _, row in Win_df.iterrows():
            r = pd.DataFrame([row])
            RWin = pd.concat([RWin, r], ignore_index=True)
            win_rally = Get_top4_states(RWin, state)
        f2_list = Plot_top_win_states(win_rally, dest_folder, state)

        for _, row in Lose_df.iterrows():
            r = pd.DataFrame([row])
            RLose = pd.concat([RLose, r], ignore_index=True)
            lose_rally = Get_top4_states(RLose, state)
        f3_list = Plot_top_lose_states(lose_rally, dest_folder, state)

        file_dict[player] = {'ShotType': f1_list, 'top_win_states': f2_list, 'top_lose_states': f3_list}

    return file_dict


def main(File_list, state):
    """
    格式:File = [file 1, file 2,...file n]
    file n = {player 1: {plot1_1: [path_list], plot1_2: [path_list], ...}, 
              player 2: {plot2_1: [path_list], plot2_2: [path_list], ...}}
    [path_list]: 假如全部的rally長度為 n, 則 path_list為 [rally1~1, rally1~2, rally1~3, ..., rally1~n], len長度為 n
    """
    File = []
    for file in File_list:
        match = pd.read_csv(file)
        file_dict = Visualize(match, state)
        File.append(file_dict)

    return File
    

if __name__ == "__main__":
    File_list = ['./input_data/all_dataset.csv']

    # 'opponent_type', 'player_location_area', 'opponent_location_area', 'hit_area'
    state = (True, True, False, True)
    main(File_list, state)