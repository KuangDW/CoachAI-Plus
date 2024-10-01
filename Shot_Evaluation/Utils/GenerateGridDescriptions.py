# Function to map the grid positions (i, j) to descriptive terms (left, mid, right / front, center, back)
def map_grid_position_to_description(grid_pos):
    if grid_pos is None:
        return "Unknown"
    
    horizontal_labels = ["left", "mid", "right"]
    vertical_labels = ["front", "center", "back"]
    
    i, j = grid_pos
    return f"{vertical_labels[j]}-{horizontal_labels[i]}"

# Generate descriptions based on the extremes for each shot type (singles court)
def generate_grid_descriptions_singles(extremes):
    description = ""

    for shot_type, data in extremes.items():
        description += f"For the {shot_type} shot type in singles court:\n"
        description += (
            f" - The highest win rate of {data['max_win_rate']:.2%} was in the {map_grid_position_to_description(data['max_win_grid'])} zone.\n"
            f" - The highest error rate of {data['max_error_rate']:.2%} occurred in the {map_grid_position_to_description(data['max_error_grid'])} zone.\n"
            #f" - The most frequent landing area was in the {map_grid_position_to_description(data['max_freq_grid'])} zone, with {data['max_freq']} shots landing in this area.\n"
        )
        description += "\n"
    
    return description