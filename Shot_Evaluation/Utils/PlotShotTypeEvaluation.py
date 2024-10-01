import os
import pandas as pd
import matplotlib.pyplot as plt
from .DrawCourt import draw_half_court
from .AnalyzeShotsbyGrid import analyze_shots_by_grid_singles
from .AnalyzeShotsbyGrid import find_extremes_in_grid_singles
from .GenerateGridDescriptions import generate_grid_descriptions_singles
import uuid

def plot_shot_type_evaluation(df, player_name, dest_folder='./Shot_Evaluation/Result'):
    os.makedirs(dest_folder, exist_ok=True)

    # Filter by the player name
    df = df[df['player'] == player_name]

    # Scale the landing positions
    df.loc[:, 'scaled_landing_x'] = df['landing_x'] * 177.5 + 177.5
    df.loc[:, 'scaled_landing_y'] = 240 - df['landing_y'] * 240

    # Get point player info
    winning_shots = df[df['getpoint_player'] == player_name].groupby(['match_id', 'set', 'rally']).tail(1)

    # Get unique shot types
    shot_types = df['type'].unique()
    excluded_types = ["Serve short", "Serve long"]
    shot_types = [shot_type for shot_type in shot_types if shot_type not in excluded_types]

    # Store the image IDs
    image_ids = []

    shot_stats = []
    # Plot the distributions for each shot type and save each as a separate image
    for shot_type in shot_types:
        shot_df = df[df['type'] == shot_type]
        total_shots = len(shot_df)
        out_df = shot_df[shot_df['lose_reason'] == 'Out of bound']
        total_outs = len(out_df)
        net_df = shot_df[shot_df['lose_reason'] == 'Net']
        total_nets = len(net_df)
        win_df = winning_shots[winning_shots['type'] == shot_type]
        total_wins = len(win_df)

        error_rate = (total_outs + total_nets) / total_shots if total_shots > 0 else 0
        win_rate = total_wins / total_shots if total_shots > 0 else 0
        shot_stats.append({
            'shot_type': shot_type,
            'total_shots': total_shots,
            'win_rate': win_rate,
            'error_rate': error_rate,
            'total_wins': total_wins,
            'total_errors': total_outs + total_nets
        })

        # Create a new figure for each shot type
        fig, ax = plt.subplots(figsize=(4, 6))

        # Draw the court on the subplot
        draw_half_court(ax=ax)

        # Plot individual landing points
        ax.scatter(out_df['scaled_landing_x'], out_df['scaled_landing_y'], c='green', label='Out', alpha=0.5)
        ax.scatter(net_df['scaled_landing_x'], net_df['scaled_landing_y'], c='red', label='Net', alpha=0.5)
        ax.scatter(win_df['scaled_landing_x'], win_df['scaled_landing_y'], c='blue', label='Win', alpha=0.5)

        # Add title and legend
        ax.set_title(f'{shot_type}\nError Rate: {total_outs + total_nets}/{total_shots}({error_rate:.2%})\nWin Rate: {total_wins}/{total_shots}({win_rate:.2%})')
        ax.legend(loc='upper right')

        # Save each plot as a separate image
        shot_id = uuid.uuid4()
        filename = f'{dest_folder}/{shot_id}.png'
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()  # Close the figure to free memory

        # Append the ID to the list
        image_ids.append(str(shot_id))

    # Grid statistics and descriptions
    '''grid_stats_singles = analyze_shots_by_grid_singles(shot_types, df, winning_shots, player_name)
    extremes_singles = find_extremes_in_grid_singles(grid_stats_singles)

    # Modify grid description generation to filter out grids with no shots
    filtered_extremes_singles = {shot_type: data for shot_type, data in extremes_singles.items()
                                 if data['max_win_rate'] > 0 and data['max_freq'] > 0}  # Filter out empty grids

    grid_description_singles = generate_grid_descriptions_singles(filtered_extremes_singles)'''

    return image_ids
