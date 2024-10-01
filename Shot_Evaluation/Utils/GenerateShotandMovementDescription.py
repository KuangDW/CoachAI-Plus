import pandas as pd
import numpy as np

# Define the court grid (6x6 grid for simplicity)
grid_x = [0, 92.5, 135, 177.5, 220, 262.5, 355]  # Updated width boundaries
grid_y = [0, 55, 110, 165, 220, 275, 480]   # Updated length boundaries

# Define movement grid for scaled positions
grid_moving_x = np.linspace(50, 305, 7)  # Horizontal divisions for movements
grid_moving_y = [0, 110, 220, 480]       # Vertical divisions for movements

def transform_coordinates_upper_court(row, player_name):
    if row['player'] == player_name:

        row['scaled_landing_x'] = row['landing_x'] * 177.5 + 177.5
        row['scaled_landing_y'] = 240 - row['landing_y'] * 240
        row['scaled_moving_x'] = row['moving_x'] * 177.5 + 177.5
        row['scaled_moving_y'] = 240 - row['moving_y'] * 240

    return row

# Function to translate grid position (x_idx, y_idx) to a detailed human-readable format
def describe_detailed_grid_position(x_idx, y_idx):
    horizontal_labels = ["Left", "Middle", "Right"]
    vertical_labels = ["Front", "Center", "Back"]
    
    x_description = horizontal_labels[x_idx // 2]  # Dividing by 2 to map 0-1 to "Left", 2-3 to "Middle", 4-5 to "Right"
    y_description = vertical_labels[y_idx // 2]  # Dividing by 2 to map 0-1 to "Front", 2-3 to "Center", 4-5 to "Back"
    description = y_description + "-" + x_description
    
    # Further refine the description with proximity to lines and specific areas
    if x_idx == 0:
        description += " near the left side line"
    elif x_idx == 5:
        description += " near the right side line"
    
    if y_idx == 0:
        description += " at the net"
    elif y_idx == 5:
        description += " near the baseline"
    elif y_idx == 1 or y_idx == 2:
        description += " near the service line"
    
    return f"{description}"

# Function to generate text descriptions for grids based on win rate and shot distribution
def generate_shot_grid_analysis_descriptions(shot_types, player_name, last_shots):
    descriptions = []

    for shot_type in shot_types:
        shot_df = last_shots[last_shots['type'] == shot_type]
        
        # Calculate the win rate grid and total count grid for this shot type
        win_count_grid = np.zeros((6, 6))
        total_count_grid = np.zeros((6, 6))

        # Assign shots to grid cells and calculate win/loss counts
        for _, row in shot_df.iterrows():
            x_idx = np.digitize(row['scaled_landing_x'], grid_x) - 1
            y_idx = np.digitize(row['scaled_landing_y'], grid_y) - 1

            if 0 <= x_idx < 6 and 0 <= y_idx < 6:  # Check bounds
                total_count_grid[x_idx, y_idx] += 1
                if row['getpoint_player'] == player_name:  # Winning shot
                    win_count_grid[x_idx, y_idx] += 1

        # Calculate win rate grid
        total_shots = total_count_grid.sum()
        threshold = max(1, 0.05 * total_shots)
        valid_grids = total_count_grid > threshold

        win_rate_grid = np.divide(win_count_grid, total_count_grid, 
                                  out=np.zeros_like(win_count_grid), where=valid_grids)

        # Find the highest win rate > 60% (valid grids only)
        if np.any(valid_grids):
            highest_win_rate_output = None
            lowest_win_rate_output = None

            # Mask invalid grids and set them to NaN
            masked_win_rate_grid = np.where(valid_grids, win_rate_grid, np.nan)

            # Find the max win rate and its position
            if np.nanmax(masked_win_rate_grid) > 0.6:
                max_win_rate = np.nanmax(masked_win_rate_grid)
                max_win_position = np.unravel_index(np.nanargmax(masked_win_rate_grid), win_rate_grid.shape)
                readable_max_win_position = describe_detailed_grid_position(*max_win_position)
                highest_win_rate_output = (
                    f"Highest win rate: {max_win_rate:.2%} in the {readable_max_win_position}\n"
                )

            # Find the min win rate and its position
            if np.nanmin(masked_win_rate_grid) < 0.4:
                min_win_rate = np.nanmin(masked_win_rate_grid)
                min_win_position = np.unravel_index(np.nanargmin(masked_win_rate_grid), win_rate_grid.shape)
                readable_min_win_position = describe_detailed_grid_position(*min_win_position)
                lowest_win_rate_output = (
                    f"Lowest win rate: {min_win_rate:.2%} in the {readable_min_win_position}\n"
                )

            # Generate description for this shot type
            description = f""
            if highest_win_rate_output:
                description += highest_win_rate_output
            if lowest_win_rate_output:
                description += lowest_win_rate_output
            if not highest_win_rate_output and not lowest_win_rate_output:
                description += "No grids met the win rate or shot threshold criteria.\n"

            descriptions.append(description)

    return descriptions


# Function to generate movement grid analysis descriptions
def generate_movement_grid_analysis_descriptions(shot_types, player_name, last_shots):
    descriptions = []

    for shot_type in shot_types:
        shot_df = last_shots[last_shots['type'] == shot_type]
        
        # Calculate the win rate grid and total count grid for this shot type
        win_count_grid = np.zeros((6, 6))
        total_count_grid = np.zeros((6, 6))

        # Assign shots to grid cells and calculate win/loss counts based on movement coordinates
        for _, row in shot_df.iterrows():
            x_idx = np.digitize(row['scaled_moving_x'], grid_moving_x) - 1
            y_idx = np.digitize(row['scaled_moving_y'], grid_moving_y) - 1

            if 0 <= x_idx < 6 and 0 <= y_idx < 6:  # Check bounds
                total_count_grid[y_idx, x_idx] += 1
                if row['getpoint_player'] == player_name:  # Winning shot
                    win_count_grid[y_idx, x_idx] += 1

        # Calculate win rate grid for movements
        total_shots = total_count_grid.sum()
        threshold = 0.05 * total_shots  # 2.5% threshold
        win_rate_grid = np.divide(win_count_grid, total_count_grid, out=np.zeros_like(win_count_grid), where=total_count_grid > threshold)

        # Only consider grids with enough data (more than 2.5% of the total points)
        valid_grids = total_count_grid > threshold

        highest_win_rate_output = None
        lowest_win_rate_output = None

        # Find the highest win rate > 60%
        if np.any(valid_grids):
            valid_high_win_grids = np.logical_and(win_rate_grid > 0.6, valid_grids)
            if np.any(valid_high_win_grids):
                max_win_rate = np.max(win_rate_grid[valid_high_win_grids])
                max_win_position = np.unravel_index(np.argmax(win_rate_grid == max_win_rate), win_rate_grid.shape)
                readable_max_win_position = describe_detailed_grid_position(*max_win_position)
                highest_win_rate_output = (
                    f"Highest win rate: {max_win_rate:.2%} for movements in the {readable_max_win_position}\n"
                )

            # Find the lowest win rate < 40%
            valid_low_win_grids = np.logical_and(win_rate_grid < 0.4, valid_grids)
            if np.any(valid_low_win_grids):
                min_win_rate = np.min(win_rate_grid[valid_low_win_grids])
                min_win_position = np.unravel_index(np.argmin(win_rate_grid == min_win_rate), win_rate_grid.shape)
                readable_min_win_position = describe_detailed_grid_position(*min_win_position)
                lowest_win_rate_output = (
                    f"Lowest win rate: {min_win_rate:.2%} for movements in the {readable_min_win_position}\n"
                )
        
        # Generate the final description based on the conditions
        description = f""
        if highest_win_rate_output:
            description += highest_win_rate_output
        if lowest_win_rate_output:
            description += lowest_win_rate_output
        if not highest_win_rate_output and not lowest_win_rate_output:
            description += "No grids met the win rate or shot threshold criteria for movements.\n"

        descriptions.append(description)

    return descriptions

def generate_distribution_grid_analysis_descriptions(shot_types, player_name, last_shots):
    descriptions = []

    for shot_type in shot_types:
        shot_df = last_shots[last_shots['type'] == shot_type]
        
        # Separate winning and losing shots for each shot type
        win_shots = shot_df[shot_df['getpoint_player'] == player_name]
        lose_shots = shot_df[shot_df['getpoint_player'] != player_name]
        
        # Calculate the mean and variance for winning and losing movement positions
        mean_win_x = win_shots['scaled_moving_x'].mean()
        mean_win_y = win_shots['scaled_moving_y'].mean()
        var_win_x = win_shots['scaled_moving_x'].var()
        var_win_y = win_shots['scaled_moving_y'].var()

        mean_lose_x = lose_shots['scaled_moving_x'].mean()
        mean_lose_y = lose_shots['scaled_moving_y'].mean()
        var_lose_x = lose_shots['scaled_moving_x'].var()
        var_lose_y = lose_shots['scaled_moving_y'].var()

        # Print mean and variance for each shot type
        description = f""
        #description += f"Winning Mean position (X, Y): ({mean_win_x:.2f}, {mean_win_y:.2f})\n"
        #description += f"Winning Variance (X, Y): ({var_win_x:.2f}, {var_win_y:.2f})\n"
        #description += f"Losing Mean position (X, Y): ({mean_lose_x:.2f}, {mean_lose_y:.2f})\n"
        #description += f"Losing Variance (X, Y): ({var_lose_x:.2f}, {var_lose_y:.2f})\n"
        
        # Compare winning and losing mean positions
        if mean_win_x > mean_lose_x:
            description += f"Winning shots are further to the right"
        else:
            description += f"Winning shots are further to the left"
        
        if mean_win_y > mean_lose_y:
            description += f" and further to the back.\n"
        else:
            description += f" and closer to the net.\n"

        # Compare variance between winning and losing
        if var_win_x > var_lose_x:
            description += f"Winning shots have a wider spread horizontally.\n"
        else:
            description += f"Losing shots have a wider spread horizontally.\n"
        
        if var_win_y > var_lose_y:
            description += f"Winning shots have a wider spread vertically.\n"
        else:
            description += f"Losing shots have a wider spread vertically.\n"
        
        descriptions.append(description)
    
    return descriptions
