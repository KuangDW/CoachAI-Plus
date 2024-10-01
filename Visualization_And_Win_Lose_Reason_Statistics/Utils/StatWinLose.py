import pandas as pd
import os
import matplotlib.pyplot as plt
import uuid

def Tally_win_loss(df, player):
    # 紀錄贏球和輸球的 row
    win_rows = []
    loss_rows = []
    # 迭代 dataframe，檢查每一個 rally 的結束和下一個開始
    for i in range(1, len(df)):
        if df['rally'].iloc[i] != df['rally'].iloc[i-1]:
            last_rally_row = df.iloc[i-1]
            current_rally_row = df.iloc[i]
            
            # 判斷前一個回合的 player 和當前回合的 player 是否相同
            if last_rally_row['player'] == current_rally_row['player']:
                if last_rally_row['player'] == player:
                    win_rows.append(last_rally_row)
                else:
                    loss_rows.append(last_rally_row)
            else:
                if last_rally_row['player'] == player:
                    loss_rows.append(last_rally_row)
                else:
                    win_rows.append(last_rally_row)

    win_df = pd.DataFrame(win_rows)
    loss_df = pd.DataFrame(loss_rows)
    
    return win_df, loss_df

# Create a badminton court layout
def draw_full_court(ax=None, color='black'):
    if ax is None:
        ax = plt.gca()
    # Draw outer lines
    ax.plot([50, 50], [150, 810], color=color)
    ax.plot([305, 305], [150, 810], color=color)
    ax.plot([27.4, 327.6], [810, 810], color=color)
    ax.plot([27.4, 327.6], [480, 480], color=color)
    ax.plot([27.4, 327.6], [150, 150], color=color)
    ax.plot([27.4, 27.4], [150, 810], color=color)
    ax.plot([327.6, 327.6], [150, 810], color=color)
    # Draw the middle line
    ax.plot([177.5, 177.5], [150, 810], color=color)
    # Draw the service lines
    ax.plot([27.4, 327.6], [594, 594], color=color)
    ax.plot([27.4, 327.6], [756, 756], color=color)
    ax.plot([27.4, 327.6], [366, 366], color=color)
    ax.plot([27.4, 327.6], [204, 204], color=color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 355)
    ax.set_ylim(0, 960)
    ax.set_aspect(1)


# Function to calculate top 4 states and return statistics dataframe
def Get_top4_states(df, state):
    cols_to_consider = []
    opponent_type, player_location_area, opponent_location_area, hit_area = state

    if opponent_type:
        cols_to_consider.append('opponent_type')
    if player_location_area:
        cols_to_consider.append('player_location_area')
    if opponent_location_area:
        cols_to_consider.append('opponent_location_area')
    if hit_area:
        cols_to_consider.append('hit_area')

    location_columns = [
        'opponent_location_x', 'opponent_location_y', 
        'player_location_x', 'player_location_y', 
        'hit_x', 'hit_y', 
        'landing_x', 'landing_y', 'type'
    ]

    cols_to_consider.extend(location_columns)

    # 統計所有狀態的數量（根據所選欄位進行分組）
    all_state_counts = df.groupby(cols_to_consider).size()

    # 統計前 4 名的狀態數量
    top4_state_counts = all_state_counts.nlargest(4)
    stats_df = top4_state_counts.reset_index()
    stats_df['top4_num'] = stats_df[0]
    stats_df = stats_df.drop(0, axis=1)
    stats_df['all_num'] = all_state_counts.sum()

    return stats_df

def generate_title(state, remaining_values):
    """根據 state 的 True/False 狀態和 remaining_values 生成標題"""
    state_names = ["opponent_type", "player_location_area", "opponent_location_area", "hit_area", "player type"]
    title_elements = []

    remaining_idx = 0
    for idx, (state_name, is_true) in enumerate(zip(state_names, state)):
        if is_true:
            title_elements.append(f"{state_name}: {remaining_values[remaining_idx]}")
            remaining_idx += 1

    return ', '.join(title_elements)

def Plot_top_win_states(state_df, dest_folder, state):
    Filename = []
    
    # 根據 all_num 分組
    grouped_stats = state_df.groupby('all_num')

    for all_num, group in grouped_stats:
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        Title = []
        first = True
        for state_tuple in group.iloc[:, :-2].to_numpy():
            # 使用 state_tuple 最後9個值來進行繪圖
            opponent_location_x, opponent_location_y, player_location_x, player_location_y, hit_x, hit_y, landing_x, landing_y, type = state_tuple[-9:]
            remaining_values = list(state_tuple[:-9]) + [state_tuple[-1]]
            title = generate_title(state, remaining_values)
            Title.append(title)
            
            # 座標轉換
            opponent_location_x = opponent_location_x * 177.5 + 177.5
            opponent_location_y = 720 - opponent_location_y * 240
            player_location_x = 177.5 - player_location_x * 177.5
            player_location_y = 240 + player_location_y * 240
            hit_x = 177.5 - hit_x * 177.5
            hit_y = 240 + hit_y * 240
            landing_x = landing_x * 177.5 + 177.5
            landing_y = 720 - landing_y * 240
            
            ### 狀態圖 ###
            draw_full_court(ax=axs[0])
            axs[0].scatter(opponent_location_x, opponent_location_y, c='red', label='Opponent', alpha=0.5)
            axs[0].scatter(player_location_x, player_location_y, c='green', label='Player', alpha=0.5)
            axs[0].scatter(hit_x, hit_y, c='blue', label='Hit Area', alpha=0.5)
            if first:
                axs[0].legend(loc='upper right', fontsize=8)
                axs[0].set_title("State")
            
            ### 動作圖 ###
            draw_full_court(ax=axs[1])
            axs[1].scatter(hit_x, hit_y, c='blue', label='Hit Area', alpha=0.5)
            axs[1].scatter(landing_x, landing_y, c='orange', label='Landing Area', alpha=0.5)
            if first:
                axs[1].legend(loc='upper right', fontsize=8)
                axs[1].set_title("Action")
                first = False

        # 標題中包含剩餘欄位的資訊
        title_str = "\n".join(Title)
        plt.suptitle(f'Win State: ( \n{title_str} )', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 儲存繪製的圖形
        id = uuid.uuid4()
        filename = f'{dest_folder}/{id}.png'
        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1)
        plt.close()

        Filename.append(id)

    return Filename


def Plot_top_lose_states(state_df, dest_folder, state):
    Filename = []
    
    # 根據 all_num 分組
    grouped_stats = state_df.groupby('all_num')

    for all_num, group in grouped_stats:
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        Title = []
        first = True
        for state_tuple in group.iloc[:, :-2].to_numpy():
            # 使用 state_tuple 最後9個值來進行繪圖
            opponent_location_x, opponent_location_y, player_location_x, player_location_y, hit_x, hit_y, landing_x, landing_y, type = state_tuple[-9:]
            remaining_values = list(state_tuple[:-9]) + [state_tuple[-1]]
            title = generate_title(state, remaining_values)
            Title.append(title)
            
            # 座標轉換
            opponent_location_x = opponent_location_x * 177.5 + 177.5
            opponent_location_y = 720 - opponent_location_y * 240
            player_location_x = 177.5 - player_location_x * 177.5
            player_location_y = 240 + player_location_y * 240
            hit_x = 177.5 - hit_x * 177.5
            hit_y = 240 + hit_y * 240
            landing_x = landing_x * 177.5 + 177.5
            landing_y = 720 - landing_y * 240
            
            ### 狀態圖 ###
            draw_full_court(ax=axs[0])
            axs[0].scatter(opponent_location_x, opponent_location_y, c='red', label='Opponent', alpha=0.5)
            axs[0].scatter(player_location_x, player_location_y, c='green', label='Player', alpha=0.5)
            axs[0].scatter(hit_x, hit_y, c='blue', label='Hit Area', alpha=0.5)
            if first:
                axs[0].legend(loc='upper right', fontsize=8)
                axs[0].set_title("State")
            
            ### 動作圖 ###
            draw_full_court(ax=axs[1])
            axs[1].scatter(hit_x, hit_y, c='blue', label='Hit Area', alpha=0.5)
            axs[1].scatter(landing_x, landing_y, c='orange', label='Landing Area', alpha=0.5)
            if first:
                axs[1].legend(loc='upper right', fontsize=8)
                axs[1].set_title("Action")
                first = False

        # 標題中包含剩餘欄位的資訊
        title_str = "\n".join(Title)
        plt.suptitle(f'Loss State: ( \n{title_str} )', fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 儲存繪製的圖形
        id = uuid.uuid4()
        filename = f'{dest_folder}/{id}.png'
        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.1)
        plt.close()

        Filename.append(id)

    return Filename