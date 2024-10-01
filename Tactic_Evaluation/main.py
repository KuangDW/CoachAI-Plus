import pandas as pd
import numpy as np
from Utils.TacticDataprocessing import ReadDataFrame, segment_dataframe, translate_coord
from Utils.Tactic import *
from Utils.Plot import *


def Layer_tacitc(rally, rally_two, rally_one, player_name, Df_tactic):
    if rally_one == []:
        # return 'No_tactic', Df_tactic
        return 'No_tactic', Df_tactic, 'No_tactic', 'No_tactic', 'No_tactic' 
    
    # layer 1
    tactic = analyze_rally_tactics(rally_two, rally_one, player_name)[0]
    tactic_l2 = analyze_rally_tactics_l2(rally_two, rally_one, player_name)[0]
    tactic_l3 = analyze_rally_tactics_l3(rally_two, rally_one, player_name)[0]
    tactic_l4 = analyze_rally_tactics_l4(rally_two, rally_one, player_name)[0]

    # # 將含有戰術的 rally 存成 csv
    if tactic != 'No_tactic':
        rally['tactic'] = tactic
        Df_tactic = pd.concat([Df_tactic, rally], ignore_index=True)
        return tactic, Df_tactic, tactic_l2, tactic_l3, tactic_l4
        # return tactic, Df_tactic
    
    # # layer 2
    # else:
    #     tactic = analyze_rally_tactics_l2(rally_two, rally_one, player_name)[0]
    #     if tactic != 'No_tactic':
    #         rally['tactic'] = tactic
    #         Df_tactic = pd.concat([Df_tactic, rally], ignore_index=True)
    #         return tactic, Df_tactic
        
    #     # layer 3    
    #     else:
    #         tactic = analyze_rally_tactics_l3(rally_two, rally_one, player_name)[0]
    #         if tactic != 'No_tactic':
    #             rally['tactic'] = tactic
    #             Df_tactic = pd.concat([Df_tactic, rally], ignore_index=True)
    #             return tactic, Df_tactic
            
    #         # layer 4 等待修改
    #         else:
    #             tactic = analyze_rally_tactics_l4(rally_two, rally_one, player_name)[0]
    #             if tactic != 'No_tactic':
    #                 rally['tactic'] = tactic
    #                 Df_tactic = pd.concat([Df_tactic, rally], ignore_index=True)
    #                 return tactic, Df_tactic
                
    return tactic, Df_tactic, tactic_l2, tactic_l3, tactic_l4
    # return tactic, Df_tactic


def Tactic(match, dest_folder = './Tactic_Evaluation/Result'):
    os.makedirs(dest_folder, exist_ok=True)

    # data processing
    df, player_list = ReadDataFrame(match)
    df_list = segment_dataframe(df)
    tactic_record_list = []
    tactic_win_record_list = []
    # file_dict = {}
    file_dict = {player: {'pie_chart': '', player: {}} for player in player_list}

    for player_name in player_list:
        # 統計 df 和dict
        Df_tactic = pd.DataFrame()    
        tactic_record = {'Full_Court_Pressure': 0, 'Defensive_Counterattack': 0, 'Four_Corner': 0, 'Forehand_Lock': 0, 
                         'Backhand_Lock': 0, 'FrontCourt_Lock': 0, 'BackCourt_Lock': 0, 'Four_Corners_Clear_Drop': 0, 'No_tactic': 0}
        tactic_win_record = {'Full_Court_Pressure': 0, 'Defensive_Counterattack': 0, 'Four_Corner': 0, 'Forehand_Lock': 0, 
                         'Backhand_Lock': 0, 'FrontCourt_Lock': 0, 'BackCourt_Lock': 0, 'Four_Corners_Clear_Drop': 0, 'No_tactic': 0}

        for rally_df in df_list:
            rally = translate_coord(rally_df, player_list)
            rally_one = rally[rally['player'] == player_name].values.tolist()
            winner = rally['getpoint_player'][0]
            rally_two = rally.values.tolist()
            
            if rally_one != []:
                tactic, Df_tactic, tactic_l2, tactic_l3, tactic_l4 = Layer_tacitc(rally, rally_two, rally_one, player_name, Df_tactic)
                # print(rally_df['getpoint_player'])
                # print(player_name, winner)
            
                # 將戰術判別結果統計起來
                tactic_record[tactic] += 1
                if(tactic_l2) != 'No_tactic':
                    tactic_record[tactic_l2] += 1
                if(tactic_l3) != 'No_tactic':
                    tactic_record[tactic_l3] += 1
                if(tactic_l4) != 'No_tactic':
                    tactic_record[tactic_l4] += 1
                if winner == player_name:
                    tactic_win_record[tactic] +=1
                    if(tactic_l2) != 'No_tactic':
                        tactic_win_record[tactic_l2] += 1
                    if(tactic_l3) != 'No_tactic':
                        tactic_win_record[tactic_l3] += 1
                    if(tactic_l4) != 'No_tactic':
                        tactic_win_record[tactic_l4] += 1


        tactic_record_list.append([tactic_record, player_name])
        tactic_win_record_list.append([tactic_win_record, player_name])
        # print(player_name,'\n', tactic_win_record)
        f = Plot_pie_chart(tactic_record, player_name, dest_folder)
        Df_tactic.to_csv(f'{dest_folder}/{player_name}_tactic.csv')
        file_dict[player_name]['pie_chart'] = f

    
    # print(tactic_win_record_list)
    for i, playerA in enumerate(player_list):
        for j, playerB in enumerate(player_list):
            if (i<j):
                f2 = Plot_histogram(tactic_record_list, playerA, playerB, dest_folder)
                f3 = coord_diagram(tactic_record_list, tactic_win_record_list, playerA, playerB, dest_folder)
                file_dict[playerA][playerB] = {'histogram': f2, 'coord_diagram': f3}
                file_dict[playerB][playerA] = {'histogram': f2, 'coord_diagram': f3}

    return file_dict


def main(File_list):
    """
    格式:File = [file 1, file 2,...file n]
    file n = {player 1: {plot1_1: path1_1, plot1_2: path1_2, ...}, 
              player 2: {plot2_1: path2_1, plot2_2: path2_2, ...}}
    """
    """
    少鈞更動：
    格式： file_dict = {player1: {'pie_chart': [uuid], player2: {'histogram': uuid, 'coord_diagram': uuid}, player3...}
                       player2: {'pie_chart': [uuid], player2: {'histogram': uuid, 'coord_diagram': uuid}, player3...}
                       .
                       .
                       .
                       }
    """

    File = []
    for file in File_list:
        match = pd.read_csv(file)
        file_dict = Tactic(match)
        File.append(file_dict)
    
    print(File)
    return File

if __name__ == "__main__":
    File_list = ['./input_data/all_dataset.csv']
    main(File_list)