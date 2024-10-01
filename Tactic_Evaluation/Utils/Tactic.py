# input: a 2-indexed list which the inner list contains only 2 element [[a,b],[c,d],[e,f]]
# output: a 2-indexed list
# 這個函式會將區間集整理成最簡區間集(最小距離小於等於1的集合將被合併)
# union_intervals([[1,3],[2,4],[5,6]]) = [[1,6]]
def union_intervals(intervals):
    
    if not intervals:
        return []

    # Step 1: Flatten the nested intervals
    interval_pieces = [item for sublist in intervals if sublist is not None for item in sublist]

    if not interval_pieces:
        return []

    # Step 2: Sort intervals by the start point
    interval_pieces.sort(key=lambda x: x[0])

    # Step 3: Initialize result with the first interval
    merged = [interval_pieces[0]]

    # Step 4: Iterate through intervals and merge them if necessary
    for current in interval_pieces[1:]:
        last_merged = merged[-1]

        # If the current interval overlaps or is adjacent to the last merged one
        if current[0] <= last_merged[1] + 1 :
            # Merge the intervals by updating the end of the last merged interval
            last_merged[1] = max(last_merged[1], current[1])
        else:
            # No overlap, add the current interval to the result
            merged.append(current)
    merged.sort(key=lambda x: x[1])    

    return merged

# input: a 2-indexed list which the inner list contains only 2 element [[a,b],[c,d],[e,f]]
# output: a integer represents the total length of intervals and the maximum integer in the intervals
# sum_and_max([[1, 6], [8, 9]) = 8, 9
def sum_and_max(intervals):
    length = 0
    max_b = 0

    for interval in intervals:
        a,b = interval
        length += (b-a+1)
        if b > max_b:
            max_b = b
    
    return length, max_b


# 戰術分析函式
# output: a 2-indexed list which the inner list contains only 2 element [[a,b],[c,d],[e,f]]
# 此函式會輸出給定rally中符合全面下壓的區間
def Full_Court_Pressure(rally):
    """判斷全面下壓"""
    
    intervals = []
    
    actions = [action[3] for action in rally]  # 假設rally中的每個action都是一個列表，第四個元素是動作類型
    n = len(actions)
    prefix_scores = [0] * (n + 1)
    
    # Calculate prefix scores based on action types
    for i in range(n):
        action = actions[i]
        score = 1 if action in ['Slice', 'Net Shot'] else 2 if action == 'Smash' else 0
        prefix_scores[i + 1] = prefix_scores[i] + score

    # Define target scores for sequences of 2, 3, and 4 shots
    targets = {2: 3, 3: 4, 4: 5}
    
    # Check for sequences of length 2 to 4
    for length in targets:
        for i in range(n - length + 1):
            score = prefix_scores[i + length] - prefix_scores[i]
            if score >= targets[length]:
                # Store the interval as 1-indexed positions
                intervals.append([i + 1, i + length])

    intervals = [intervals]

    intervals = union_intervals(intervals)

    return intervals if intervals != [] else None

# output: a 2-indexed list which the inner list contains only 2 element [[a,b],[c,d],[e,f]]
# 此函式會輸出給定rally中符合四角調動的區間
def Four_Corners_Clear_Drop(rally):
    """四角調動判斷"""
    
    scores = []
    previous_location = rally[0][5]

    if previous_location in [1, 2, 5, 31, 32]:
        previous_location = 6
    elif previous_location in [3, 4, 8, 27, 28]:
        previous_location = 5
    elif previous_location in [6, 9, 10, 14, 30]:
        previous_location = 4
    elif previous_location in [7, 11, 12, 15, 26]:
        previous_location = 3
    elif previous_location in [13, 17, 18, 21, 22, 29]:
        previous_location = 2
    elif previous_location in [16, 19, 20, 23, 24, 25]:
        previous_location = 1
    else:
        previous_location = 0
    
    for action in rally[1:]:
        hit_location = action[5]
        if hit_location in [1, 2, 5, 31, 32]:
            hit_location = 6
        elif hit_location in [3, 4, 8, 27, 28]:
            hit_location = 5
        elif hit_location in [6, 9, 10, 14, 30]:
            hit_location = 4
        elif hit_location in [7, 11, 12, 15, 26]:
            hit_location = 3
        elif hit_location in [13, 17, 18, 21, 22, 29]:
            hit_location = 2
        elif hit_location in [16, 19, 20, 23, 24, 25]:
            hit_location = 1
        else:
            hit_location = 0

        if hit_location != previous_location and hit_location in [1, 2, 5, 6]:
            scores.append(1)
        else:
            scores.append(0)
        previous_location = hit_location

    n = len(scores)
    if n < 4:
        return None

    # Calculate prefix sums
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + scores[i - 1]

    intervals = []

    # Check for all intervals of length 4 with score sum >= 3
    for i in range(n - 3):
        if prefix_sum[i + 4] - prefix_sum[i] >= 3:
            intervals.append([i + 1, i + 4])

    intervals = [intervals]
    intervals = union_intervals(intervals)

    return intervals if intervals != [] else None

# output: a 2-indexed list which the inner list contains only 2 element [[a,b],[c,d],[e,f]]
# 此函式會輸出給定rally中符合正手鎖定的區間
def Forehand_Lock(rally):
    """判斷正手鎖定戰術"""
    scores = []
    for action in rally:
        hit_location = action[5]
        if hit_location in [27, 28, 3, 4, 8, 7, 11, 12, 15, 16, 26]:
            scores.append(1)
        else:
            scores.append(0)

    n = len(scores)
    if n < 4:
        return None

    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + scores[i - 1]

    intervals = []
    for i in range(n - 3):
        if prefix_sum[i + 4] - prefix_sum[i] >= 2:
            intervals.append([i + 1, i + 4])

    intervals = [intervals]
    intervals = union_intervals(intervals)

    return intervals if intervals != [] else None

# output: a 2-indexed list which the inner list contains only 2 element [[a,b],[c,d],[e,f]]
# 此函式會輸出給定rally中符合反手鎖定的區間
def Backhand_Lock(rally):
    """判斷反手鎖定戰術"""
    scores = []
    for action in rally:
        hit_location = action[5]
        if hit_location in [1, 2, 31, 32, 5, 6, 9, 10, 30, 13, 14]:
            scores.append(1)
        else:
            scores.append(0)

    n = len(scores)
    if n < 4:
        return None

    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + scores[i - 1]

    intervals = []

    for i in range(n - 3):
        if (prefix_sum[i + 4] - prefix_sum[i] >= 2 and scores[i] > 0 and scores[i + 3] > 0):
            intervals.append([i + 1, i + 4])

    intervals = [intervals]
    intervals = union_intervals(intervals)

    return intervals if intervals !=[] else None
    
# output: a 2-indexed list which the inner list contains only 2 element [[a,b],[c,d],[e,f]]
# 此函式會輸出給定rally中符合前場鎖定的區間
def FrontCourt_Lock(rally):
    """判斷前場鎖定"""
    scores = []
    for action in rally:
        hit_location = action[5]
        if hit_location in [29, 21, 22, 23, 24, 25, 17, 18, 19, 20]:
            scores.append(1)
        else:
            scores.append(0)

    n = len(scores)
    if n < 3:
        return None

    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + scores[i - 1]

    intervals = []

    for i in range(n - 2):
        if (prefix_sum[i + 3] - prefix_sum[i] >= 2 and scores[i] > 0 and scores[i + 2] > 0):
            intervals.append([i + 1, i + 3])

    intervals = [intervals]
    intervals = union_intervals(intervals)

    return intervals if intervals !=[] else None

# output: a 2-indexed list which the inner list contains only 2 element [[a,b],[c,d],[e,f]]
# 此函式會輸出給定rally中符合後場鎖定的區間
def BackCourt_Lock(rally):
    """判斷後場鎖定"""
    scores = []

    for action in rally:
        hit_location = action[5]
        hit_type = action[3]
        if hit_location in [31, 32, 28, 27, 1, 2, 3, 4, 5, 6, 7, 8] and hit_type != 'Smash':
            scores.append(1)
        else:
            scores.append(0)

    n = len(scores)
    if n < 4:
        return None

    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + scores[i - 1]

    intervals = []

    for i in range(n - 3):
        if (prefix_sum[i + 4] - prefix_sum[i] >= 2 and scores[i] > 0 and scores[i + 3] > 0):
            intervals.append([i + 1, i + 4])

    intervals = [intervals]
    intervals = union_intervals(intervals)

    return intervals if intervals !=[] else None

# output: a 2-indexed list which the inner list contains only 2 element [[a,b],[c,d],[e,f]]
# 此函式會輸出給定rally中符合四角拉吊的區間
def Four_Corner(rally):
    ctrb = BackCourt_Lock(rally)
    ctrf = FrontCourt_Lock(rally)
    ctrr = Forehand_Lock(rally)
    ctrl = Backhand_Lock(rally)
    fccd = Four_Corners_Clear_Drop(rally)

    if(ctrr == None and ctrl == None and ctrf == None and ctrb == None and fccd == None):
        return None
    
    ctr = union_intervals([ctrr, ctrl, ctrf, ctrb, fccd])

    return ctr

# output: a list of two elements: [e,f]
# 此函式會輸出給定rally中符合守中反攻的最後一個區間
def Defensive_Counterattack(rally, player_name):
    """判斷守中反攻：對方殺球後進入防守，直到我方殺球"""
    in_defense_mode = False
    defense_periods = []
    start_index = None

    for i, action in enumerate(rally):
        player = action[2]
        shot_type = action[3]

        if shot_type == 'Smash' and player != player_name:
            # 對方殺球時，我方進入防守模式
            if not in_defense_mode:
                in_defense_mode = True
                start_index = i+1  # 記錄開始防守的位置

        # 如果我方在防守模式並且我方執行殺球
        if in_defense_mode and (shot_type == 'Smash' or shot_type == 'Slice' or shot_type=='Net Shot') and player == player_name:
            # 防守模式結束，記錄防守期間
            if start_index is not None:
                defense_periods.append([start_index, i+1])
                start_index = None
            in_defense_mode = False
    
    # print(rally)
    # print('defense:\n', defense_periods)
    # 選擇最長或最晚開始的防守期間
    if defense_periods:  # 檢查是否有防守記錄
        defense_periods.sort(key=lambda x: -x[1], reverse=True)
        defense_periods = defense_periods[0]
    else:
        defense_periods = None

    return defense_periods


# 戰術統整主函式
# 此函式會回傳層次一的戰術
# Output: ['No_tactic'], ['Defensive_Counterattack'], ['Full_Court_Pressure'], ['Four_Corner'] 其中之一
def analyze_rally_tactics(rally_two, rally, player):
    t_num = 0
    ball_num = {}
    
    if Full_Court_Pressure(rally) is not None:
        ball_num['Full_Court_Pressure'] = Full_Court_Pressure(rally)
        t_num += 1
    if Defensive_Counterattack(rally_two, player) is not None:
        ball_num['Defensive_Counterattack'] = Defensive_Counterattack(rally_two, player)
        t_num += 1
    if Four_Corner(rally) is not None:
        ball_num['Four_Corner'] = Four_Corner(rally)
        t_num += 1

    # 若沒有檢驗到戰術則返回 no tactic
    if t_num == 0:
        return ['No_tactic']
    
    # 檢查最後兩個動作
    last_two_actions = rally[-2:]

    # step1: 若倒數兩顆有守中反攻
    if 'Defensive_Counterattack' in ball_num and any(
            last_two_actions[i][1] in range(ball_num['Defensive_Counterattack'][0], ball_num['Defensive_Counterattack'][1] + 1)
            for i in range(2)):
        return ['Defensive_Counterattack']
    
    # step2: 如果區間只有一個元素，直接選擇該戰術
    for key, interval in ball_num.items():
        if len(ball_num) == 1:
            return [key]
     
    # step3: 檢查全面下壓跟四角拉吊
    if 'Full_Court_Pressure' in ball_num:
        Full_Court_Pressure_len, full_court_pressure_b = sum_and_max(ball_num['Full_Court_Pressure'])
    else: 
        Full_Court_Pressure_len, full_court_pressure_b = [0, 0]
    if 'Four_Corner' in ball_num:
        Four_Corner_len, Four_Corner_b = sum_and_max(ball_num['Four_Corner'])
    else:
        Four_Corner_len, Four_Corner_b = [0, 0]

    if full_court_pressure_b > Four_Corner_b:
        return ['Full_Court_Pressure']
    if full_court_pressure_b < Four_Corner_b:
        return ['Four_Corner']

    # step3: 若倒數兩顆有無殺球
    last_two_smash = any(action[3] == 'Smash' for action in last_two_actions)
    # print('last_two_actions:\n',last_two_actions)

    if last_two_smash and 'Full_Court_Pressure' in ball_num:
        # Check if any of the last two actions fall within the range specified in 'Full_Court_Pressure'
        a = ball_num['Full_Court_Pressure'][-1][0]
        a = 2 * a - 1
        b = ball_num['Full_Court_Pressure'][-1][1]
        b = 2 * b + 1
        if any(last_two_actions[i][1] in range(a, b) for i in range(2)):
            return ['Full_Court_Pressure']
    else:
        return ['Four_Corner']

    

# 層次二
# 此函式會回傳層次二的戰術
# Output: ['No_tactic'], ['Forehand_Lock'], ['Backhand_Lock'] 其中之一
def analyze_rally_tactics_l2(rally_two, rally, player):
    t_num = 0
    ball_num = {}
    Forehand_Lock_len, Forehand_Lock_b, Backhand_Lock_len, Backhand_Lock_b = [0, 0, 0, 0]
    
    if Forehand_Lock(rally) is not None:
        ball_num['Forehand_Lock'] = Forehand_Lock(rally)
        Forehand_Lock_len, Forehand_Lock_b = sum_and_max(ball_num['Forehand_Lock'])
        t_num += 1
    if Backhand_Lock(rally) is not None:
        ball_num['Backhand_Lock'] = Backhand_Lock(rally)
        Backhand_Lock_len, Backhand_Lock_b = sum_and_max(ball_num['Backhand_Lock'])
        t_num += 1

    # 若沒有檢驗到戰術則返回 no tactic
    if t_num == 0:
        return ['No_tactic']

    if Forehand_Lock_len > Backhand_Lock_len:
        return ['Forehand_Lock']
    if Forehand_Lock_len < Backhand_Lock_len:
        return ['Backhand_Lock']
    
    if Forehand_Lock_b > Backhand_Lock_b:
        return ['Forehand_Lock']
    
    return ['Backhand_Lock']

# 層次三
# 此函式會回傳層次三的戰術
# Output: ['No_tactic'], ['FrontCourt_Lock'], ['BackCourt_Lock'] 其中之一
def analyze_rally_tactics_l3(rally_two, rally, player):
    t_num = 0
    ball_num = {}

    FrontCourt_Lock_len, FrontCourt_Lock_b, BackCourt_Lock_len, BackCourt_Lock_b = [0, 0, 0, 0]
    
    if FrontCourt_Lock(rally) is not None:
        ball_num['FrontCourt_Lock'] = FrontCourt_Lock(rally)
        FrontCourt_Lock_len, FrontCourt_Lock_b = sum_and_max(ball_num['FrontCourt_Lock'])
        t_num += 1
    if BackCourt_Lock(rally) is not None:
        ball_num['BackCourt_Lock'] = BackCourt_Lock(rally)
        BackCourt_Lock_len, BackCourt_Lock_b = sum_and_max(ball_num['BackCourt_Lock'])
        t_num += 1

    # 若沒有檢驗到戰術則返回 no tactic
    if t_num == 0:
        return ['No_tactic']
    
    if FrontCourt_Lock_len > BackCourt_Lock_len:
        return ['FrontCourt_Lock']
    if FrontCourt_Lock_len < BackCourt_Lock_len:
        return ['BackCourt_Lock']
    
    if FrontCourt_Lock_b > BackCourt_Lock_b:
        return ['FrontCourt_Lock']
    
    return ['BackCourt_Lock']

# 層次四
# 此函式會回傳層次四的戰術
# Output: ['No_tactic'], ['Four_Corners_Clear_Drop'] 其中之一
def analyze_rally_tactics_l4(rally_two, rally, player):
    t_num = 0
    ball_num = {}
    
    # 求擊球數
    actions = [action[3] for action in rally]
    n = len(actions)

    
    if Four_Corners_Clear_Drop(rally) is not None:
        ball_num['Four_Corners_Clear_Drop'] = Four_Corners_Clear_Drop(rally)
        t_num += 1

    # 若沒有檢驗到戰術則返回 no tactic
    if t_num == 0:
        return ['No_tactic']
    
    Four_Corners_Clear_Drop_len, Four_Corners_Clear_Drop_b = sum_and_max(ball_num['Four_Corners_Clear_Drop'])

    if (n - Four_Corners_Clear_Drop_b) < 3 and ( ((analyze_rally_tactics_l2 == ['No_tactic']) and (analyze_rally_tactics_l3 == ['No_tactic'])) or ((analyze_rally_tactics_l2 != ['No_tactic']) and (analyze_rally_tactics_l3 != ['No_tactic']))):
        return ['Four_Corners_Clear_Drop']
    
    return ['No_tactic']