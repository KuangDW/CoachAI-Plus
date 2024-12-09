import os

import numpy as np
import pandas as pd
from .Utils.coord2area import coord2area
from .Utils.Dataprocess import *
from .Utils.PlotLastBallRound import plot_last_ball_round
from .Utils.PlotShotTypeEvaluation import plot_shot_type_evaluation
from .Utils.PlotTopReasons import plot_top_reasons
from .Utils.PlotDensityDifferenceBetweenMatches import plot_density_difference_between_matches
from .Utils.CalculateLastBallRoundJSD import calculate_last_ball_round_jsd

from .shot_influence import plot_shot_influence
from .exertion import plot_energetic_cost


def Evaluation(match, last_ball_round = 3, dest_folder = './Shot_Evaluation/Result',matchId1 = 23, matchId2 = 28):
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


    #6. 最後n球 分布比較
    if matchId1 in df['match_id'].values and matchId2 in df['match_id'].values:
        fn10 = None#plot_density_difference_between_matches(df, player_list[0], matchId1, matchId2, last_ball_round, dest_folder)
    else:
        fn10 = None


    js = calculate_last_ball_round_jsd(df, player_list[0], player_list[1], last_ball_round, dest_folder)


    result = {
        player_list[0]: {'shot_type': fn1, 'last_ball': fn3, 'top_reasons': fn5, 'shot_influence': fn7},
        player_list[1]: {'shot_type': fn2, 'last_ball': fn4, 'top_reasons': fn6, 'shot_influence': fn8},
        "energetic_cost": fn9
    }

    if fn10 is not None:
        result["density_difference"] = fn10

    return result