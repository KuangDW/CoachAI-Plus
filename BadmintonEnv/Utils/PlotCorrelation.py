import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

# read file
def read_file(fileReal):
    match_Real = pd.read_csv(fileReal)
    match_Real = DataPrepocessing(match_Real)
    return match_Real

def DataPrepocessing(df):
    type = {'發短球': 1.0, '長球': 2.0, '推撲球': 3.0, 
            '殺球': 4.0, '接殺防守': 5.0, '平球': 6.0, 
            '接不到': 11.0, '網前球': 7.0, '挑球': 8.0, 
            '切球': 9.0, '發長球': 10.0, '接不到': 11.0}
    
    df['hit_area_state'] = df['hit_area']
    df['hit_area_action'] = df['hit_area']

    df['type'] = df['type'].map(type)
    df_way = df[['rally', 'ball_round' ,'player', 'type', 'opponent_type',
                'player_location_area', 'hit_area_state', 'opponent_location_area', 'player_move_area', 'landing_area', 'hit_area_action']].copy()    
    #df_way = df_way[df_way['type'] != 'Missed shot'].reset_index(drop=True)

    # rename
    renamed_columns = {
        'opponent_type': 'State 0',
        'player_location_area': 'State 1',
        'opponent_location_area': 'State 2',
        'hit_area_state': 'State 3',
        'type': 'Action 0',
        'hit_area_action': 'Action 1',
        'landing_area': 'Action 2',
        'player_move_area': 'Action 3'
    }
    df_way.rename(columns=renamed_columns, inplace=True)

    # combine
    threshold = 50
    for col in ['State 0', 'State 1', 'State 2', 'State 3', 
                'Action 0', 'Action 1', 'Action 2', 'Action 3']:
        counts = df_way[col].value_counts()
        to_replace = counts[counts < threshold].index
        df_way[col] = df_way[col].replace(to_replace, '其他')
    
    df_way['State0_State3'] = df_way['State 0'].astype(str) + "_" + df_way['State 3'].astype(str)
    df_way['Action0_State1'] = df_way['Action 0'].astype(str) + "_" + df_way['State 1'].astype(str)

    return df_way


# calculate Cramer's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.size == 0:
        return np.nan
    chi2, p, dof, ex = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    if min(rcorr-1, kcorr-1) == 0:
        return np.nan
    return np.sqrt(phi2corr / min(kcorr-1, rcorr-1))


# calculate and plot Cramer's V heatmap
def Plot_cramersv_heatmap(df):
    """state_columns = ['State 0', 'State 1', 'State 2', 'State 3']
    action_columns = ['Action 0', 'Action 1', 'Action 2', 'Action 3']"""

    state_columns = ['State0_State3', 'State 1', 'Action0_State1']
    action_columns = ['Action 0', 'Action 1', 'Action 2']

    all_columns = state_columns + action_columns

    # init
    cramersv_matrix = pd.DataFrame(np.zeros((len(all_columns), len(all_columns))), 
                                   index=all_columns, columns=all_columns)
    
    for col1 in all_columns:
        for col2 in all_columns:
            if col1 == col2:
                cramersv_matrix.loc[col1, col2] = 1.0
            elif col2 in cramersv_matrix.columns[:cramersv_matrix.columns.get_loc(col1)]:
                # 由於矩陣是對稱的，只計算一次
                cramersv_matrix.loc[col1, col2] = cramersv_matrix.loc[col2, col1]
            else:
                cramersv_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

    plt.figure(figsize=(12, 10))
    sns.heatmap(cramersv_matrix, annot=True, cmap="Blues", linewidths=0.5, fmt='.2f')
    plt.title("Cramer's V Correlation Heatmap between Categorical Variables")
    plt.show()


# conditional Cramer's V correlation
def calculate_specific_cramersv(df):
    # 1. based on State 0 and State 3 -> Action 0
    cramersv1 = cramers_v(df['State0_State3'], df['Action 0'])
    
    # 2. based on Action 0 and State 1 -> Action 2
    cramersv2 = cramers_v(df['Action0_State1'], df['Action 2'])
    
    # 3. based on State 1 -> Action 1
    cramersv3 = cramers_v(df['State 1'], df['Action 1'])

    return cramersv1, cramersv2, cramersv3



if __name__ == "__main__":
    file_path = './input_data/All_dataset.csv'
    df = read_file(file_path)
    Plot_cramersv_heatmap(df)
    cramersv1, cramersv2, cramersv3 = calculate_specific_cramersv(df)
    print(f"Cramer's V between (State 0 + State 3) and Action 0: {cramersv1:.2f}")
    print(f"Cramer's V between (Action 0 + State 1) and Action 2: {cramersv2:.2f}")
    print(f"Cramer's V between State 1 and Action 1: {cramersv3:.2f}")