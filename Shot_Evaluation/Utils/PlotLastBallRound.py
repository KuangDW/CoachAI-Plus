from .CoordinatesProcess import transform_coordinates
import pandas as pd
import os
import matplotlib.pyplot as plt
from .DrawCourt import draw_full_court
import seaborn as sns
import matplotlib.patches as mpatches
from .GenerateShotandMovementDescription import *
import uuid

def plot_last_ball_round(df, player_name, last_n, dest_folder='./Shot_Evaluation/Result'):
    os.makedirs(dest_folder, exist_ok=True)
    
    # Filter by the player name
    df = df[df['player'] == player_name]

    # Scale the landing positions
    df = df.apply(transform_coordinates, axis=1, args=(player_name,))
    
    # Filter winning and losing shots
    winning_shots = df[df['getpoint_player'] == player_name].groupby(['match_id', 'set', 'rally']).tail(last_n)
    losing_shots = df[df['getpoint_player'] != player_name].groupby(['match_id', 'set', 'rally']).tail(last_n)

    # Get unique shot types
    shot_types = df['type'].unique()
    excluded_types = ["Serve short", "Serve long"]
    shot_types = [shot_type for shot_type in shot_types if shot_type not in excluded_types]

    # Dictionary to store image filenames for each shot type
    image_ids = {}

    # Plot each shot type separately
    for shot_type in shot_types:
        # Create a new figure for each shot type
        fig, ax = plt.subplots(figsize=(4, 6))

        # Filter for specific shot types
        win_df = winning_shots[winning_shots['type'] == shot_type]
        lose_df = losing_shots[losing_shots['type'] == shot_type]

        # Draw the court on the subplot
        draw_full_court(ax=ax)

        # Plot heat maps using KDE (Kernel Density Estimate)
        sns.kdeplot(x=win_df['scaled_landing_x'], y=win_df['scaled_landing_y'], ax=ax, cmap='Greens', fill=True, alpha=0.5, label='Winning Rallies')
        sns.kdeplot(x=lose_df['scaled_landing_x'], y=lose_df['scaled_landing_y'], ax=ax, cmap='Reds', fill=False, alpha=0.5, label='Losing Rallies')
        
        sns.kdeplot(x=win_df['scaled_moving_x'], y=win_df['scaled_moving_y'], ax=ax, cmap='Greens', fill=True, alpha=0.5)
        sns.kdeplot(x=lose_df['scaled_moving_x'], y=lose_df['scaled_moving_y'], ax=ax, cmap='Reds', fill=False, alpha=0.5)

        # Add title
        ax.set_title(f'{shot_type}', fontsize=14)

        # Add legends for colors
        green_patch = mpatches.Patch(color='green', label='Win (Landing/Movement)')
        red_patch = mpatches.Patch(color='red', label='Lose (Landing/Movement)')
        #fig.legend(handles=[green_patch, red_patch], loc='lower center', ncol=2, fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Save the plot for each shot type
        shot_id = uuid.uuid4()
        filename = f'{dest_folder}/{shot_id}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        # Store the image filename for each shot type
        image_ids[shot_type] = shot_id
        
        '''# Upper court data for descriptions
        df_upper_court = df.apply(transform_coordinates_upper_court, axis=1, args=(player_name,))
        winning_shots_upper_court = df_upper_court[df['getpoint_player'] == 'opponent'].groupby(['match_id', 'set', 'rally']).tail(last_n)
        losing_shots_upper_court = df_upper_court[df['getpoint_player'] == 'player'].groupby(['match_id', 'set', 'rally']).tail(last_n)
        last_shots_upper_court = pd.concat([winning_shots_upper_court, losing_shots_upper_court])

        # Generate descriptions
        landing_descriptions = generate_shot_grid_analysis_descriptions(shot_types, player_name, last_shots_upper_court)
        movement_descriptions = generate_movement_grid_analysis_descriptions(shot_types, player_name, last_shots_upper_court)
        distribution_descriptions = generate_distribution_grid_analysis_descriptions(shot_types, player_name, last_shots_upper_court)

        # Save descriptions to txt files
        with open(f'./Shot_Evaluation/Result/landing_descriptions_{player_name}.txt', 'w') as f:
            for shot_type, desc in zip(shot_types, landing_descriptions):
                f.write(f"Shot Type: {shot_type}\n")
                f.write(desc + "\n\n")

        with open(f'./Shot_Evaluation/Result/movement_descriptions_{player_name}.txt', 'w') as f:
            for shot_type, desc in zip(shot_types, movement_descriptions):
                f.write(f"Shot Type: {shot_type}\n")
                f.write(desc + "\n\n")
        
        with open(f'./Shot_Evaluation/Result/distribution_descriptions_{player_name}.txt', 'w') as f:
            for shot_type, desc in zip(shot_types, distribution_descriptions):
                f.write(f"Shot Type: {shot_type}\n")
                f.write(desc + "\n\n")'''

    return image_ids
