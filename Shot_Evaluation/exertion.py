# usage: calculate_exertion -> plot_energetic_cost
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import uuid


#nuumerator: [pace of player's shot ("player's"), pace of incoming shot (default)]
#denominator: [player move to hit ("after"), player move after hit (default)]
def calculate_exertion(df_Real, numerator = "opponent's", denominator = "after"):
    df_Real['pace'] = np.nan
    df_Real['exertion'] = np.nan
    for index, _ in df_Real.iterrows():
        if  pd.isna(df_Real.loc[index, "lose_reason"]) and df_Real.loc[index, "server"] != 3:
            fly_x = df_Real.loc[index,'hit_x']+df_Real.loc[index,'landing_x']
            fly_y = 2-(df_Real.loc[index,'hit_y']+df_Real.loc[index,'landing_y']) #modified mid: +1 -> 1-hit_y + 1-landing_y
            fly_distance = (fly_x**2 + fly_y**2) ** 0.5
            df_Real.loc[index,'fly_distance'] = fly_distance
            fly_time = df_Real.loc[index+1, "frame_num"] - df_Real.loc[index, "frame_num"]
            if fly_time > 0:
                df_Real.loc[index,"pace"] = fly_distance/fly_time

        if  df_Real.loc[index, "server"] == 2:
            if denominator == "after":
                move_x = df_Real.loc[index,'player_move_x']
                move_y = df_Real.loc[index,'player_move_y']
                move_time = df_Real.loc[index+1, "frame_num"] - df_Real.loc[index, "frame_num"]
            elif denominator == "before":
                move_x = df_Real.loc[index-1,'opponent_move_x']
                move_y = df_Real.loc[index-1,'opponent_move_y']
                move_time = df_Real.loc[index, "frame_num"] - df_Real.loc[index-1, "frame_num"]
            move_distance = (move_x**2 + move_y**2) ** 0.5
            if move_time > 0:
                if move_distance > 0:
                    if numerator == "player's":
                        exertion = df_Real.loc[index,"pace"]/(move_distance/move_time)
                    elif numerator == "opponent's":
                        exertion = df_Real.loc[index-1,"pace"]/(move_distance/move_time)
                    df_Real.loc[index,"exertion"] = exertion
        
    # Normalize pace and exertion
    scaler = MinMaxScaler()
    df_Real[['pace', 'exertion']] = scaler.fit_transform(df_Real[['pace', 'exertion']])
    df_Real['exertion'] = df_Real['exertion'].fillna(0)

    # Aggregate exertion per rally and player
    exertion_per_rally = df_Real.groupby(['rally_id', 'player'])['exertion'].sum().reset_index()
    exertion_per_rally.rename(columns={'exertion': 'exertion_per_rally'}, inplace=True)
    df_Real = df_Real.merge(exertion_per_rally, on=['rally_id', 'player'], how='left')

    return df_Real


def plot_energetic_cost(df_Real, match_id, set_num, player_name, dest_folder = './Shot_Evaluation/Result'):
    os.makedirs(dest_folder, exist_ok=True)

    filtered_data = df_Real[(df_Real['match_id'] == match_id) & (df_Real['set'] == set_num)]
    players = filtered_data['player'].unique()
    opponent = players[0] if players[0] != player_name else players[1]

    # Extract exertion per rally for each player
    player_A_data = filtered_data[filtered_data['player'] == player_name][['rally', 'exertion_per_rally']]
    player_B_data = filtered_data[filtered_data['player'] == opponent][['rally', 'exertion_per_rally']]

    # Group by rally_id and calculate the mean exertion per rally for each player
    player_A_exertion = player_A_data.groupby('rally')['exertion_per_rally'].mean().reset_index()
    player_B_exertion = player_B_data.groupby('rally')['exertion_per_rally'].mean().reset_index()

    # Extract ball rounds per rally
    ball_rounds_data = filtered_data.groupby('rally')['ball_round'].max().reset_index()
    getpoint_player_data = filtered_data.groupby('rally')['getpoint_player'].last().reset_index()

    # Merge ball rounds data with getpoint player data
    ball_rounds_data = ball_rounds_data.merge(getpoint_player_data, on='rally')

    # Plotting
    fig, ax1 = plt.subplots(figsize=(15, 7))

    ax1.plot(player_A_exertion['rally'], player_A_exertion['exertion_per_rally'], label=f'{player_name}', color='darkorange', marker='o')
    ax1.plot(player_B_exertion['rally'], player_B_exertion['exertion_per_rally'], label=f'{opponent}', color='darkblue', marker='o')

    # Adding labels and title
    ax1.set_xlabel('Rally')
    ax1.set_ylabel('Energetic Cost')
    ax1.set_title(f'Energetic Cost Per Rally and Ball Rounds Distribution of Set{set_num}')
    ax1.legend(loc='upper left')

    # Define colors for players
    colors = {f'{player_name}': 'gold', f'{opponent}': 'cornflowerblue'}

    # Creating a second y-axis to plot the ball rounds distribution
    ax2 = ax1.twinx()
    bars = ax2.bar(ball_rounds_data['rally'], ball_rounds_data['ball_round'], 
                color=[colors[player] for player in ball_rounds_data['getpoint_player']], 
                alpha=0.3, label='Ball Rounds')

    # Adding label for the second y-axis
    ax2.set_ylabel('Number of Ball Rounds')

    # Adding a legend for the bars
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='gold', lw=4, label=f'{player_name} scores'),
                    Line2D([0], [0], color='cornflowerblue', lw=4, label=f'{opponent} scores')]
    ax2.legend(handles=legend_elements, loc='upper right')

    # Set x-ticks to start from 1
    rally_ids = ball_rounds_data['rally']
    ax1.set_xticks(rally_ids)
    ax1.set_xticklabels(rally_ids)

    # Show the plot
    id = uuid.uuid4()
    filename = f'{dest_folder}/{id}.png'
    plt.savefig(filename)
    plt.close()

    return id
