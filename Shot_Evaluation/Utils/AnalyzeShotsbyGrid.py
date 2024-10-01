import numpy as np

# Function to assign shots to a 3x3 grid based on their scaled landing positions (singles court)
def assign_to_grid_singles(scaled_x, scaled_y):
    # Inner singles court x-coordinates (0 to 355), y-coordinates (0 to 480)
    grid_x = [0, 135, 220, 355]  # Updated width boundaries
    grid_y = [0, 110, 220, 480]   # Updated length boundaries

    # Determine which grid the shot falls into based on x and y positions
    for i in range(3):
        for j in range(3):
            if grid_x[i] <= scaled_x < grid_x[i+1] and grid_y[j] <= scaled_y < grid_y[j+1]:
                return (i, j)
    
    return None  # Return None if the shot is outside the expected range

# Function to analyze shot performance by grid for singles court
def analyze_shots_by_grid_singles(shot_types, dataset, winning_shots, player_name):
    grid_stats = {}
    
    for shot_type in shot_types:
        shot_df = dataset[dataset['type'] == shot_type]
        
        # Initialize grid for win and error counts
        grid = np.zeros((3, 3), dtype={'names': ('wins', 'errors', 'total'), 'formats': ('i4', 'i4', 'i4')})

        for _, row in shot_df.iterrows():
            grid_pos = assign_to_grid_singles(row['scaled_landing_x'], row['scaled_landing_y'])
            if grid_pos:
                grid[grid_pos]['total'] += 1
                
                # Check if the shot resulted in a win or error
                if row['getpoint_player'] == player_name:
                    grid[grid_pos]['wins'] += 1
                elif row['lose_reason'] in ['Out of bound', 'Net']:
                    grid[grid_pos]['errors'] += 1
        
        grid_stats[shot_type] = grid
    
    return grid_stats

# Function to find extremes in each grid (singles court)
def find_extremes_in_grid_singles(grid_stats):
    extremes = {}

    for shot_type, grid in grid_stats.items():
        max_win_rate = 0
        max_error_rate = 0
        max_freq = 0
        max_win_grid = None
        max_error_grid = None
        max_freq_grid = None

        for i in range(3):
            for j in range(3):
                total_shots = grid[i, j]['total']
                if total_shots > 0:
                    win_rate = grid[i, j]['wins'] / total_shots
                    error_rate = grid[i, j]['errors'] / total_shots
                    freq = total_shots

                    # Check for highest win rate
                    if win_rate > max_win_rate:
                        max_win_rate = win_rate
                        max_win_grid = (i, j)

                    # Check for highest error rate
                    if error_rate > max_error_rate:
                        max_error_rate = error_rate
                        max_error_grid = (i, j)

                    # Check for most frequent landing zone
                    if freq > max_freq:
                        max_freq = freq
                        max_freq_grid = (i, j)

        # Store the extremes for this shot type
        extremes[shot_type] = {
            'max_win_rate': max_win_rate,
            'max_win_grid': max_win_grid,
            'max_error_rate': max_error_rate,
            'max_error_grid': max_error_grid,
            'max_freq': max_freq,
            'max_freq_grid': max_freq_grid
        }

    return extremes
