import os, sys
import pandas as pd

# Add paths
ROOT_FILDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_FILDER)

from Tactic_Evaluation import Tactic

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