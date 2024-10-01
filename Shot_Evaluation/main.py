import os

import numpy as np
import pandas as pd
from Utils.coord2area import coord2area
from Utils.Dataprocess import *
from Utils.PlotLastBallRound import plot_last_ball_round
from Utils.PlotShotTypeEvaluation import plot_shot_type_evaluation
from Utils.PlotTopReasons import plot_top_reasons

from shot_influence import plot_shot_influence
from exertion import plot_energetic_cost


def Evaluation(match, last_ball_round = 3, dest_folder = './Shot_Evaluation/Result'):
    """所對應分別是：
    df: dataframe
    df_dict: df長度統計dict
    df_List: df根據rally切分dataframe的list
    df_len: df的rally數量
    """

    df, df_dict, df_List, df_len = ReadDataFrame(match)
    player_list = df["player"].unique()
    print("Players:", player_list)
    
    os.makedirs(dest_folder, exist_ok=True)

    # player_list正常來說會有兩個人，因此可以兩個人都分析
    fileModel1 = f"./Shot_Evaluation/win_prob_data/dataset_{player_list[0]}.csv"
    fileModel2 = f"./Shot_Evaluation/win_prob_data/dataset_{player_list[1]}.csv"
    df_model1 = pd.read_csv(fileModel1)
    df_model2 = pd.read_csv(fileModel2)

    # 1. 選手的位置＆球的落點分佈
    fn1 = plot_shot_type_evaluation(df, player_list[0], dest_folder)
    fn2 = plot_shot_type_evaluation(df, player_list[1], dest_folder)
    print("Plot location distribution")


    # 2. 最後n球 選手的位置＆球的落點分佈
    fn3 = plot_last_ball_round(df, player_list[0], last_ball_round, dest_folder)
    fn4 = plot_last_ball_round(df, player_list[1], last_ball_round, dest_folder)
    print("Plot last 3 ball rounds distribution")


    # 3. 贏球輸球的原因
    fn5 = plot_top_reasons(df_List, player_list[0], dest_folder)
    fn6 = plot_top_reasons(df_List, player_list[1], dest_folder)
    print("Top 3 reasons for winning/losing")
    

    # 4. Shot influence
    fn7 = []
    fn8 = []
    match_id = df.iloc[0]["match_id"]
    set_num = df.iloc[0]["set"]
    # Original
    # rallies = df["rally"].unique()
    # Updated
    testDf = df[df["match_id"] == match_id]
    testDf = testDf[testDf["set"] == set_num]
    rallies = testDf["rally"].unique()
    
    for i in rallies:
        fn7.append(plot_shot_influence.main(player_list[0], match_id, set_num, i, df, df_model1, dest_folder))
    for i in rallies:
        fn8.append(plot_shot_influence.main(player_list[1], match_id, set_num, i, df, df_model2, dest_folder))

    #5. 體力消耗
    fn9 = plot_energetic_cost(df, match_id, set_num, player_list[0], dest_folder)

    return {player_list[0]: {'shot_type': fn1, 'last_ball': fn3, 'top_reasons': fn5, 'shot_influence': fn7},
            player_list[1]: {'shot_type': fn2, 'last_ball': fn4, 'top_reasons': fn6, 'shot_influence': fn8},
            "energetic_cost": fn9}


def main(File_list):
    """
    格式:File = [file 1, file 2,...file n]
    file n = {player 1: {plot1_1: path1_1, plot1_2: path1_2, ...}, 
              player 2: {plot2_1: path2_1, plot2_2: path2_2, ...}}
    """
    File = []
    for file in File_list:
        match = pd.read_csv(file)

        match['match_id'] = match['match_id'].astype(int).astype(str)
        match_id_mapping = {'1':'23', '3': '28', '5': '30', '6': '31', '7': '32', '13': '49', '2': '25', '4': '29', '8': '36', '9': '43', '10': '44',
                    '11': '45', '19': '55', '14': '50', '15': '51', '30': '72', '36': '79', '44': '97', '17': '53', '26': '64', '12': '48',
                    '20': '56', '23': '60', '28': '69', '41': '88', '18': '54', '16': '52', '27': '66', '33': '75', '37': '82', '24': '61',
                    '39': '86', '21': '57', '34': '76', '29': '71', '35': '78', '38': '85', '22': '58', '32': '74', '25': '63', '31': '73',
                    '43': '94', '42': '89', '40': '87'}
        match_id_mapping = {k: int(v) for k, v in match_id_mapping.items()}
        match['match_id'] = match['match_id'].map(match_id_mapping)
        
        file_dict = Evaluation(match)
        File.append(file_dict)

    return File


if __name__ == "__main__":
    File_list = ["./input_data/all_dataset.csv"]
    main(File_list)
