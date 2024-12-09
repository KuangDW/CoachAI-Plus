import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import uuid
from scipy.stats import gaussian_kde
from .DrawCourt import draw_full_court
from .CoordinatesProcess import transform_coordinates

def plot_density_difference_between_matches(df, player_name, match_id_1, match_id_2, last_n, dest_folder='./Shot_Evaluation/Result', bandwidth='scott'):
    """
    Plots only the density differences for movement and landing between two specified matches for the same player
    on the full court for winning and losing shots, using the 'RdBu' color scheme. The color scheme is reversed for losing shots.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing match data.
    - player_name (str): The name of the player.
    - match_id_1 (int): The ID of the first match to compare.
    - match_id_2 (int): The ID of the second match to compare.
    - last_n (int): The number of last shots to consider in each rally.
    - dest_folder (str): The destination folder to save results.
    - bandwidth (float or callable, optional): Bandwidth for the KDE calculation. If None, the default bandwidth is used.

    Returns:
    - image_ids (dict): A dictionary mapping shot types and results (win/lose) to image filenames.
    """
    os.makedirs(dest_folder, exist_ok=True)

    # Filter the DataFrame for the player and matches, and apply coordinate transformation
    df_match_1 = df[(df['player'] == player_name) & (df['match_id'] == match_id_1)].apply(transform_coordinates, axis=1, args=(player_name,))
    df_match_2 = df[(df['player'] == player_name) & (df['match_id'] == match_id_2)].apply(transform_coordinates, axis=1, args=(player_name,))

    # Define court boundaries
    xmin, xmax = 0, 355  # Full court width
    ymin, ymax = 0, 960  # Full court height

    # Create a grid for KDE evaluation over the entire court
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Helper function to compute KDE and density difference
    def compute_density_difference(x1, y1, x2, y2, bandwidth):
        # Perform KDE for each set of positions
        values1 = np.vstack([x1, y1])
        if values1.shape[1] < 2:
            return None
        kernel1 = gaussian_kde(values1, bw_method=bandwidth)
        Z1 = np.reshape(kernel1(positions).T, X.shape)

        values2 = np.vstack([x2, y2])
        if values2.shape[1] < 2:
            return None
        kernel2 = gaussian_kde(values2, bw_method=bandwidth)
        Z2 = np.reshape(kernel2(positions).T, X.shape)

        # Calculate density difference
        Z_diff = Z1 - Z2
        max_abs_Z_diff = np.max(np.abs(Z_diff))
        return Z_diff / max_abs_Z_diff if max_abs_Z_diff != 0 else Z_diff

    # Get unique shot types and exclude serves
    shot_types = [shot_type for shot_type in df['type'].unique() if shot_type not in {"Serve short", "Serve long"}]

    # Dictionary to store filenames
    image_ids = {}

    # Loop through each shot type and compute the density difference for winning and losing shots
    for shot_type in shot_types:
        for result in ['win', 'lose']:
            # Filter the last shots by type and result for both matches
            match_1_shots = df_match_1[(df_match_1['type'] == shot_type) & (df_match_1['getpoint_player'] == player_name if result == 'win' else df_match_1['getpoint_player'] != player_name)]
            match_2_shots = df_match_2[(df_match_2['type'] == shot_type) & (df_match_2['getpoint_player'] == player_name if result == 'win' else df_match_2['getpoint_player'] != player_name)]

            match_1_shots = match_1_shots.groupby(['set', 'rally']).tail(last_n)
            match_2_shots = match_2_shots.groupby(['set', 'rally']).tail(last_n)

            # Extract landing and movement positions
            x1_landing, y1_landing = match_1_shots['scaled_landing_x'], match_1_shots['scaled_landing_y']
            x2_landing, y2_landing = match_2_shots['scaled_landing_x'], match_2_shots['scaled_landing_y']
            x1_movement, y1_movement = match_1_shots['scaled_moving_x'], match_1_shots['scaled_moving_y']
            x2_movement, y2_movement = match_2_shots['scaled_moving_x'], match_2_shots['scaled_moving_y']

            # Compute density differences
            Z_diff_landing = compute_density_difference(x1_landing, y1_landing, x2_landing, y2_landing, bandwidth)
            Z_diff_movement = compute_density_difference(x1_movement, y1_movement, x2_movement, y2_movement, bandwidth)

            # Check if density differences were computed
            if Z_diff_landing is None and Z_diff_movement is None:
                print(f"Not enough data for density difference plots: {shot_type}, {result.capitalize()}")
                continue

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            draw_full_court(ax=ax)

            # Set colormap based on result type
            cmap = 'RdBu' if result == 'win' else 'RdBu_r'

            # Plot Landing Density Difference
            if Z_diff_landing is not None:
                c1 = ax.contourf(X, Y, Z_diff_landing, levels=20, cmap=cmap, alpha=0.6, vmin=-1, vmax=1)
                fig.colorbar(c1, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

            # Plot Movement Density Difference
            if Z_diff_movement is not None:
                c2 = ax.contourf(X, Y, Z_diff_movement, levels=20, cmap=cmap, alpha=0.3, vmin=-1, vmax=1)

            title_text = f"{shot_type} - {result.capitalize()} Shots\n{player_name} Comparison: Match {match_id_1} vs. Match {match_id_2}"
            ax.set_title(title_text, fontsize=16, pad=15)

            plt.tight_layout()

            shot_id = uuid.uuid4()
            filename = f"{dest_folder}/{shot_id}.png"
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            image_ids[f"{shot_type}_{result}"] = shot_id

    return image_ids
