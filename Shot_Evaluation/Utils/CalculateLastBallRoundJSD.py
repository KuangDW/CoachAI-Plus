import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import uuid

def calculate_last_ball_round_jsd(df, player1_name, player2_name, last_n, dest_folder='./Shot_Evaluation/Result'):
    """
    Generates KDE plots for winning and losing shots to compare shot distributions between two players.
    Additionally, calculates Jensen-Shannon Divergence (JS Divergence) for both landing and moving positions 
    by shot type in win/lose cases and overall cases, then outputs the results into a text file.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing match data.
    - player1_name (str): The name of the first player.
    - player2_name (str): The name of the second player.
    - last_n (int): The number of last shots to consider in each rally.
    - dest_folder (str): The destination folder to save results.
    
    Returns:
    - js_file_path (str): The path to the text file containing the JS divergence calculations.
    """
    os.makedirs(dest_folder, exist_ok=True)

    # Filter by the player name
    df_player1 = df[df['player'] == player1_name]
    df_player2 = df[df['player'] == player2_name]

    # Scale the landing and movement positions
    def scale_coordinates(df):
        df['scaled_landing_x'] = df['landing_x'] * 177.5 + 177.5
        df['scaled_landing_y'] = 240 - df['landing_y'] * 240
        df['scaled_moving_x'] = df['moving_x'] * 177.5 + 177.5
        df['scaled_moving_y'] = 240 - df['moving_y'] * 240
        return df

    df_player1 = scale_coordinates(df_player1)
    df_player2 = scale_coordinates(df_player2)

    def calculate_js_divergence(x1, y1, x2, y2, bins=100, epsilon=1e-10):
        hist1, _, _ = np.histogram2d(x1, y1, bins=bins, range=[[0, 355], [0, 240]], density=False)
        hist2, _, _ = np.histogram2d(x2, y2, bins=bins, range=[[0, 355], [0, 240]], density=False)

        # Flatten histograms and add epsilon to avoid zero values
        hist1 = hist1.flatten() + epsilon
        hist2 = hist2.flatten() + epsilon

        # Normalize histograms, ensuring no division by zero
        s1, s2 = np.sum(hist1), np.sum(hist2)
        if s1 > 0:
            hist1 /= s1
        if s2 > 0:
            hist2 /= s2

        # Calculate JS Divergence if neither histogram is all zero
        if np.all(hist1 == 0) or np.all(hist2 == 0):
            return float('nan')  # Return NaN if comparison is invalid
        return jensenshannon(hist1, hist2) ** 2



    # Group shots by type and calculate JSD for each type
    shot_types = df['type'].unique()
    js_divergences = {}

    for shot_type in shot_types:
        # Filter shots by type for both players
        player1_shots = df_player1[df_player1['type'] == shot_type].groupby(['match_id', 'set', 'rally']).tail(last_n)
        player2_shots = df_player2[df_player2['type'] == shot_type].groupby(['match_id', 'set', 'rally']).tail(last_n)

        # Split shots into winning and losing for both players
        player1_winning_shots = player1_shots[player1_shots['getpoint_player'] == player1_name]
        player1_losing_shots = player1_shots[player1_shots['getpoint_player'] != player1_name]
        player2_winning_shots = player2_shots[player2_shots['getpoint_player'] == player2_name]
        player2_losing_shots = player2_shots[player2_shots['getpoint_player'] != player2_name]

        # Calculate JSD for landing and movement, win/lose for each type
        js_divergences[shot_type] = {
            'landing_win': calculate_js_divergence(
                player1_winning_shots['scaled_landing_x'], player1_winning_shots['scaled_landing_y'],
                player2_winning_shots['scaled_landing_x'], player2_winning_shots['scaled_landing_y']
            ),
            'landing_lose': calculate_js_divergence(
                player1_losing_shots['scaled_landing_x'], player1_losing_shots['scaled_landing_y'],
                player2_losing_shots['scaled_landing_x'], player2_losing_shots['scaled_landing_y']
            ),
            'moving_win': calculate_js_divergence(
                player1_winning_shots['scaled_moving_x'], player1_winning_shots['scaled_moving_y'],
                player2_winning_shots['scaled_moving_x'], player2_winning_shots['scaled_moving_y']
            ),
            'moving_lose': calculate_js_divergence(
                player1_losing_shots['scaled_moving_x'], player1_losing_shots['scaled_moving_y'],
                player2_losing_shots['scaled_moving_x'], player2_losing_shots['scaled_moving_y']
            ),
            # Overall (winning + losing combined) JSD calculations for landing and movement
            'landing_overall': calculate_js_divergence(
                pd.concat([player1_winning_shots['scaled_landing_x'], player1_losing_shots['scaled_landing_x']]),
                pd.concat([player1_winning_shots['scaled_landing_y'], player1_losing_shots['scaled_landing_y']]),
                pd.concat([player2_winning_shots['scaled_landing_x'], player2_losing_shots['scaled_landing_x']]),
                pd.concat([player2_winning_shots['scaled_landing_y'], player2_losing_shots['scaled_landing_y']])
            ),
            'moving_overall': calculate_js_divergence(
                pd.concat([player1_winning_shots['scaled_moving_x'], player1_losing_shots['scaled_moving_x']]),
                pd.concat([player1_winning_shots['scaled_moving_y'], player1_losing_shots['scaled_moving_y']]),
                pd.concat([player2_winning_shots['scaled_moving_x'], player2_losing_shots['scaled_moving_x']]),
                pd.concat([player2_winning_shots['scaled_moving_y'], player2_losing_shots['scaled_moving_y']])
            )
        }

    js_df = pd.DataFrame(js_divergences)

    # Save to CSV
    js_csv_path = os.path.join(dest_folder, f'{player1_name}_vs_{player2_name}_js_divergence_by_type.csv')
    js_df.to_csv(js_csv_path, index=False)
    
    # # Write JS divergence results to a text file
    # js_file_path = os.path.join(dest_folder, f'{player1_name}_vs_{player2_name}_js_divergence_by_type.txt')
    # with open(js_file_path, 'w') as f:
    #     f.write("Jensen-Shannon Divergence by shot type between distributions:\n\n")
    #     for shot_type, js_vals in js_divergences.items():
    #         f.write(f"Shot Type: {shot_type}\n")
    #         f.write(f"  Landing (Win) JS Divergence: {js_vals['landing_win']:.5f}\n")
    #         f.write(f"  Landing (Lose) JS Divergence: {js_vals['landing_lose']:.5f}\n")
    #         f.write(f"  Moving (Win) JS Divergence: {js_vals['moving_win']:.5f}\n")
    #         f.write(f"  Moving (Lose) JS Divergence: {js_vals['moving_lose']:.5f}\n")
    #         f.write(f"  Landing (Overall) JS Divergence: {js_vals['landing_overall']:.5f}\n")
    #         f.write(f"  Moving (Overall) JS Divergence: {js_vals['moving_overall']:.5f}\n\n")

    # return js_file_path
