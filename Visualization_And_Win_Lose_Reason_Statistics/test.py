import os, sys
import pandas as pd

# Add paths
ROOT_FILDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_FILDER)

from Visualization_And_Win_Lose_Reason_Statistics import Visualize

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