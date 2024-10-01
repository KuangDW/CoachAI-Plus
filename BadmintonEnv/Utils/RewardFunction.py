def Reward_function(Violation_reason, Mode, IsConstraint):
    """Run the reward function"""
    # init
    reward = 0
    if IsConstraint:
        reward -= 1

    if Mode == 'Win':
        reward += 1

    elif Mode == 'lose':
        if Violation_reason == 'Out of bound':
            reward -= 2
        elif Violation_reason == 'Miss hit':
            reward -= 2
            
    elif Mode == 'play':
        reward += 0

    return reward