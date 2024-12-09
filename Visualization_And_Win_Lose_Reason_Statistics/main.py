import pandas as pd
import numpy as np
import os
from .Utils.ShotTypeDistribution import Player_ShotType_histogram
from .Utils.Dataprocessing import ReadDataframe, Split_dataframe, Segment_dataframe
from .Utils.StatWinLose import *


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