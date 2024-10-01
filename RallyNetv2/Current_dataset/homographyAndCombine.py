import pandas as pd
import numpy as np
import ast
import os
import re

def combine_and_homography():    
    data_folder = 'data/' #args['input_data_folder_path']
    match_list_csv = 'match.csv' #args['match_list_csv']
    homography_matrix_list_csv = 'homography.csv' #args['homography_matrix_list_csv']
    output_path = 'data_tables/raw_dataset.csv' #args['prepared_data_output_path']

    match_list = pd.read_csv(data_folder + match_list_csv)

    homography_matrix_list = pd.read_csv(data_folder + homography_matrix_list_csv, converters={'homography_matrix':lambda x: np.array(ast.literal_eval(x))})
    
    available_matches = []
    for idx in range(len(match_list)):
        match_idx = match_list['id'][idx]
        match_name = match_list['video'][idx]
        winner = match_list['winner'][idx]
        loser = match_list['loser'][idx]

        homography_matrix = homography_matrix_list[homography_matrix_list['id'] == match_idx]['homography_matrix'].to_numpy()[0]

        match_folder = os.path.join(data_folder, match_name)
        set_csv = [os.path.join(match_folder, f) for f in os.listdir(match_folder) if f.endswith('.csv')]
        
        match_data = []
        for csv in set_csv:
            set_data = pd.read_csv(csv)
            set_data['player'] = set_data['player'].replace(['A', 'B'], [winner, loser])
            set_data['set'] = re.findall(r'\d+', os.path.basename(csv))[0]
            match_data.append(set_data)

        match = pd.concat(match_data, ignore_index=True, sort=False).assign(match_id=match_idx)
        
        # project screen coordinate to real coordinate
        for i in range(len(match)):
            player_location = np.array([match['player_location_x'][i], match['player_location_y'][i], 1])
            player_location_real = homography_matrix.dot(player_location)
            player_location_real /= player_location_real[2]

            match.iloc[i, match.columns.get_loc('player_location_x')] = player_location_real[0]
            match.iloc[i, match.columns.get_loc('player_location_y')] = player_location_real[1]
            
            opponent_location = np.array([match['opponent_location_x'][i], match['opponent_location_y'][i], 1])
            opponent_location_real = homography_matrix.dot(opponent_location)
            opponent_location_real /= opponent_location_real[2]

            match.iloc[i, match.columns.get_loc('opponent_location_x')] = opponent_location_real[0]
            match.iloc[i, match.columns.get_loc('opponent_location_y')] = opponent_location_real[1]

            landing_location = np.array([match['landing_x'][i], match['landing_y'][i], 1])
            landing_location_real = homography_matrix.dot(landing_location)
            landing_location_real /= landing_location_real[2]

            match.iloc[i, match.columns.get_loc('landing_x')] = landing_location_real[0]
            match.iloc[i, match.columns.get_loc('landing_y')] = landing_location_real[1]

            hit_location = np.array([match['hit_x'][i], match['hit_y'][i], 1])
            hit_location_real = homography_matrix.dot(hit_location)
            hit_location_real /= hit_location_real[2]

            match.iloc[i, match.columns.get_loc('hit_x')] = hit_location_real[0]
            match.iloc[i, match.columns.get_loc('hit_y')] = hit_location_real[1]


        available_matches.append(match)
    
    available_matches = pd.concat(available_matches, ignore_index=True, sort=False)
    
    #cleaned_matches = preprocess_data(available_matches)
    #cleaned_matches.to_csv(args['prepared_data_output_path'], index=False)

    # 分數調整
    dataframe_temp = pd.DataFrame(columns=['value'],index=['temp'])
    for index, row in available_matches[::-1].iterrows():
        if type(available_matches['getpoint_player'][index]) == str:
            dataframe_temp.loc['temp','value']  = available_matches.loc[index, 'getpoint_player']
        who_win =  dataframe_temp.loc['temp','value']
        temp = available_matches.loc[index, 'roundscore_'+ str(who_win)]
        available_matches.loc[index, 'roundscore_'+ who_win] = temp-1
        # 故意把這一個rally的結果先顯示出來
        available_matches.loc[index, 'getpoint_player'] =  who_win

    available_matches.to_csv(output_path, index=False)

if __name__ == '__main__':
    combine_and_homography()
