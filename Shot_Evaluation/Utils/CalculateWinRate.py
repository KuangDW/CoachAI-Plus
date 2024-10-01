def calculate_win_rate(df, player_name):
    """ Calculate the average number of points scored by player A relative to the total points scored by A and B. """
    count_A = df['getpoint_player'].value_counts().get(player_name, 0)
    count_B = df['getpoint_player'].value_counts().drop(player_name, errors='ignore').sum()
    total_points = count_A + count_B
    
    if total_points == 0:
        total_count = 0
    else:
        total_count = round(count_A / total_points, 4)
    
    print(f'Player Winner Rate: {total_count}')
    

    return total_count