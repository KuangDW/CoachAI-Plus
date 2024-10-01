import os
import heapq
import pandas as pd
import matplotlib.pyplot as plt
from .DrawCourt import draw_full_court
import uuid

def coordContinuous2Discrete(x, y, position):
    if position == 1:
        pass
    elif position == 2:
        x = -x
        y = -y
    else:
        raise NotImplementedError

    if y < 0:
        return 10
    elif y < 110:
        region = [1, 2, 3]
    elif y < 220:
        region = [4, 5, 6]
    elif y < 330:
        region = [7, 8, 9]
    else:
        return 10

    if x < -127.5:
        return 10
    elif x < -42.5:
        return region[2]
    elif x < 42.5:
        return region[1]
    elif x < 127.5:
        return region[0]
    else:
        return 10

def get_row_state(dataset, player_name, result):
    if len(dataset) < 2:
        return None, None
    if result == 'lose':
        row = dataset.iloc[-1] if dataset.iloc[-1]['player'] != player_name else dataset.iloc[-2]
        player_region = coordContinuous2Discrete(row['opponent_location_x'] * 177.5, 240 - row['opponent_location_y'] * 240, 1)
        opponent_region = coordContinuous2Discrete(row['player_location_x'] * 177.5, 240 - row['player_location_y'] * 240, 1)
    else:
        row = dataset.iloc[-1] if dataset.iloc[-1]['player'] == player_name else dataset.iloc[-2]
        opponent_region = coordContinuous2Discrete(row['opponent_location_x'] * 177.5, 240 - row['opponent_location_y'] * 240, 1)
        player_region = coordContinuous2Discrete(row['player_location_x'] * 177.5, 240 - row['player_location_y'] * 240, 1)
    ball_region = coordContinuous2Discrete(row['landing_x'] * 177.5, 240 - row['landing_y'] * 240, 1)
    state = (player_region, opponent_region, ball_region)
    lose_reason = dataset.iloc[-1]['lose_reason']

    return (state, lose_reason)

def plot_top_reasons(Real_List, player_name, dest_folder = './Shot_Evaluation/Result'):
    os.makedirs(dest_folder, exist_ok=True)
    
    win_reason_count_statistic = {}
    lose_reason_count_statistic = {}
    
    for df in Real_List:
        if df.iloc[-1]['getpoint_player'] != player_name:
            state, lose_reason = get_row_state(df, player_name, 'lose')
            if (state, lose_reason) not in lose_reason_count_statistic:
                lose_reason_count_statistic[(state, lose_reason)] = 0
            lose_reason_count_statistic[(state, lose_reason)] += 1
        else:
            state, lose_reason = get_row_state(df, player_name, 'win')
            if (state, lose_reason) not in win_reason_count_statistic:
                win_reason_count_statistic[(state, lose_reason)] = 0
            win_reason_count_statistic[(state, lose_reason)] += 1

    win_top4 = heapq.nlargest(4, win_reason_count_statistic.items(), key=lambda x: x[1])
    lose_top4 = heapq.nlargest(4, lose_reason_count_statistic.items(), key=lambda x: x[1])
    win_total = sum(win_reason_count_statistic.values())
    lose_total = sum(lose_reason_count_statistic.values())

    win_shots_dict = {key: [] for key, _ in win_top4}
    lose_shots_dict = {key: [] for key, _ in lose_top4}

    for df in Real_List:
        state, lose_reason = get_row_state(df, player_name, 'lose')
        if (state, lose_reason) in lose_shots_dict:
            shot = df.iloc[-1] if df.iloc[-1]['player'] != player_name else df.iloc[-2]
            lose_shots_dict[(state, lose_reason)].append(shot)
        state, lose_reason = get_row_state(df, player_name, 'win')
        if (state, lose_reason) in win_shots_dict:
            shot = df.iloc[-1] if df.iloc[-1]['player'] == player_name else df.iloc[-2]
            win_shots_dict[(state, lose_reason)].append(shot)

    win_shots_dfs = {key: pd.DataFrame(shots) for key, shots in win_shots_dict.items()}
    lose_shots_dfs = {key: pd.DataFrame(shots) for key, shots in lose_shots_dict.items()}

    # Plotting top 4 win shots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for index, (key, df) in enumerate(win_shots_dfs.items()):
        df['scaled_landing_x'] = df['landing_x'] * 177.5 + 177.5
        df['scaled_landing_y'] = 720 - df['landing_y'] * 240
        df['scaled_opponent_location_x'] = df['opponent_location_x'] * 177.5 + 177.5
        df['scaled_opponent_location_y'] = 720 - df['opponent_location_y'] * 240
        df['scaled_player_location_x'] = 177.5 - df['player_location_x'] * 177.5
        df['scaled_player_location_y'] = 240 + df['player_location_y'] * 240

        draw_full_court(ax=axs[index])

        axs[index].scatter(df['scaled_landing_x'], df['scaled_landing_y'], c='blue', label='Ball', alpha=0.5)
        axs[index].scatter(df['scaled_opponent_location_x'], df['scaled_opponent_location_y'], c='red', label='Opponent', alpha=0.5)
        axs[index].scatter(df['scaled_player_location_x'], df['scaled_player_location_y'], c='green', label='Player', alpha=0.5)

        axs[index].set_title(f'Player: {key[0][0]} Opponent: {key[0][1]} Ball: {key[0][2]}\nWin reason: {key[1]} Percent: {"{:.1%}".format(len(df)/win_total)}', fontsize=10)
        axs[index].legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Top 4 Win Reasons - {player_name}', fontsize=16, y=0.98)
    plt.tight_layout()
    win_id = uuid.uuid4()
    filename = f'{dest_folder}/{win_id}.png'
    plt.savefig(filename)
    plt.close()

    # Plotting top 4 lose shots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for index, (key, df) in enumerate(lose_shots_dfs.items()):
        df['scaled_landing_x'] = df['landing_x'] * 177.5 + 177.5
        df['scaled_landing_y'] = 720 - df['landing_y'] * 240
        df['scaled_opponent_location_x'] = df['player_location_x'] * 177.5 + 177.5
        df['scaled_opponent_location_y'] = 720 - df['player_location_y'] * 240
        df['scaled_player_location_x'] = 177.5 - df['opponent_location_x'] * 177.5
        df['scaled_player_location_y'] = 240 + df['opponent_location_y'] * 240

        draw_full_court(ax=axs[index])

        axs[index].scatter(df['scaled_landing_x'], df['scaled_landing_y'], c='blue', label='Ball', alpha=0.5)
        axs[index].scatter(df['scaled_opponent_location_x'], df['scaled_opponent_location_y'], c='red', label='Opponent', alpha=0.5)
        axs[index].scatter(df['scaled_player_location_x'], df['scaled_player_location_y'], c='green', label='Player', alpha=0.5)

        axs[index].set_title(f'Player: {key[0][0]} Opponent: {key[0][1]} Ball: {key[0][2]}\nLose reason: {key[1]} Percent: {"{:.1%}".format(len(df)/lose_total)}', fontsize=10)
        axs[index].legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Top 4 Lose Reasons - {player_name}', fontsize=16, y=0.98)
    loose_id = uuid.uuid4()
    filename = f'{dest_folder}/{loose_id}.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return {"win": win_id, "loose": loose_id}