import pandas as pd
import numpy as np
from Utils.BaseServe import BaseServe
from Utils.Constraint import BadmintonConstraint
from Utils.RewardFunction import Reward_function

class BadmintonEnv:
    def __init__(self, player_agent, opponent_agent, player_name, opponent_name, episodes, is_match, is_constraint, is_have_serve_state, filepath):
        # input
        self.agent_a = player_agent
        self.agent_b = opponent_agent
        self.agent_a_name = player_name
        self.agent_b_name = opponent_name
        self.episodes = episodes
        self.is_match = is_match
        self.is_constraint = is_constraint
        self.is_have_serve_state = is_have_serve_state
        self.filepath = filepath

        # environment information
        self.score_a = 0
        self.score_b = 0
        self.pre_score_a = 0
        self.pre_score_b = 0
        self.rally = 0
        self.match = 1
        self.round = 1
        self.serving_player = 'A'  # A starts serving
        
        self.state = None
        self.info_dict = {}

        # function operation
        self.reward_function = Reward_function
        self.badminton_constraint = BadmintonConstraint()
        self.baseServe = BaseServe()

        # Other
        self.type_mapping = {1: 'Serve short', 2: 'Clear', 3: 'Push Shot', 4: 'Smash', 5: 'Smash Defence', 
                             6: 'Drive', 7: 'Net Shot', 8: 'Lob', 9: 'Drop', 10: 'Serve long', 11: 'Missed shot'}
        self.output = pd.DataFrame()

    # ================== Envrionment coordinates adjustion ================== #
    def flip_coordinates(self, coords):
        """Flip the coordinates for perspective change."""
        return (-coords[0], -coords[1])

    def check_out_violation(self, action, state, is_launch):
        """Check if the shot is violation"""
        if not (-127.5 <= action[2][0] <= 127.5) or not (0 <= action[2][1] <= 330):
            return 'Out of bound'
        
        if is_launch:
            # if opponent is on the x > 0, then it's 0, otherwise, it's 1
            serve_way = 0 if state[2][0] > 0 else 1
            if serve_way == 0:
                if not (0 <= action[2][0] <= 127.5) or not (144 <= action[2][1] <= 276):
                    return 'Serve Out of bound'
            else:
                if not (-127.5 <= action[2][0] <= 0) or not (144 <= action[2][1] <= 276):
                    return 'Serve Out of bound'
        return None

    # ================== Serve state adjustion ================== #
    def adjusted_serve_position(self, position, player):
        """Adjust the serving position based on the score."""
        x, y = position
        if player == 'A':
            if self.score_a % 2 == 0 and x < 0:
                x *= -1  # Even score, serve from the right side
            elif self.score_a % 2 != 0 and x > 0:
                x *= -1  # Odd score, serve from the left side

        else:
            if self.score_b % 2 == 0 and x < 0:
                x *= -1  # Even score, serve from the right side
            elif self.score_b % 2 != 0 and x > 0:
                x *= -1  # Odd score, serve from the left side
        return (x, y)
    
    def adjusted_defense_position(self, server_position, opp_position):
        """Adjust player & opponent relative position when serving"""
        x, y = opp_position
        if (server_position[0] > 0 and x > 0) or (server_position[0] < 0 and x < 0):
            x *= -1
        return (x, y)

    # ================== Environment function: reset, step, info ================== #
    def reset(self):
        self.rally += 1

        """reset serving by rules"""
        # A serve
        if self.serving_player == 'A':
            # When the agent has its own serve positioning function
            if self.is_have_serve_state:
                s = self.agent_a.serve_state()
            else:
                s = self.baseServe.serve_state()
            adjusted_self_position = self.adjusted_serve_position(s[1], 'A')

        # B serve
        else:
            # When the agent has its own serve positioning function
            if self.is_have_serve_state:
                s = self.agent_b.serve_state()
            else:
                s = self.baseServe.serve_state()
            adjusted_self_position = self.adjusted_serve_position(s[1], 'B')    

        adjusted_opp_position = self.adjusted_defense_position(adjusted_self_position, s[2])
        self.state = (s[0], adjusted_self_position, adjusted_opp_position, s[3])
        self.round = 1

        if self.serving_player == 'B':
            # False indicates that after Reset, A does not need to serve
            action_b = self.agent_b.action(self.state, self.info('B', None, self.state, violation = False, constraint = False), launch = True)
            state, info, done, launch = self.step(action_b, is_launch = True)
            return state, info, done, launch
        else:
            # True indicates that after Reset, A needs to serve
            return self.state, self.info('A', None, self.state, violation = False, constraint = False), False, True
    

    def step(self, action, is_launch):
        """
        input: A agent action, is launch
        output: state, reward, info, done, is_launch
        """

        # Scenario 1. B action check
        if self.serving_player == 'B' and is_launch == True:
            action_b = action
            violation_reason_b = None
            if action_b is None or self.check_out_violation(action_b, self.state, is_launch) is not None:
                if action_b is None:
                    violation_reason_b = 'Miss hit'
                else:
                    violation_reason_b = self.check_out_violation(action_b, self.state, is_launch)

                # Next time A will serve, update the score
                self.serving_player = 'A'
                self.score_a += 1

                # updata csv information
                self.Record('B', self.state, action_b, IsLaunch = True, IsConstraint = False, violation = violation_reason_b)
                info = self.info('B', action, self.state, violation = True, constraint = False)
                done = True
                return self.state, info, done, False
            else:
                # Check if it exceeds the constraint
                if self.is_constraint:
                    action_b, _, isConstraint_b = self.badminton_constraint.validate_action(self.state, action_b)
                else:
                    isConstraint_b = False

                # updata csv information
                self.Record('B', self.state, action_b, IsLaunch = True, IsConstraint = isConstraint_b, violation = violation_reason_b)
                info = self.info('B', action, self.state, violation = False, constraint = isConstraint_b)
                done = False

                # Then convert B's action into a state that the environment can accept and return it to A
                st = self.state[2]
                self.state = (action_b[0],  # 自身使用的球種 --> 對手使用的球種 # Own shot type --> Opponent's shot type
                            self.flip_coordinates(st), # 對手的座標(+) --> 自身的座標(-) # Opponent's coordinates (+) --> Own coordinates (-)
                            self.flip_coordinates(action_b[3]), # 自身的移動(-) --> 對手的座標(+) # Own movement (-) --> Opponent's coordinates (+)
                            self.flip_coordinates(action_b[2]))  # 自身使用的落點(+) --> 球的現在位置(-) # Own landing point (+) --> Current position of the shuttlecock (-)
            
                return self.state, info, done, False
        
        
        violation_reason_a = None         
        violation_reason_b = None

        # Scenario 2. When A commits a foul.
        if action is None or self.check_out_violation(action, self.state, is_launch) is not None:
            if action is None:
                violation_reason_a = 'Miss hit'
            else:
                violation_reason_a =  self.check_out_violation(action, self.state, is_launch)

            # Update reward, info, done
            reward = self.reward_function(Violation_reason = violation_reason_a, Mode = 'lose', IsConstraint = False)
            done = True

            # Next time B will serve, and update the score
            self.serving_player = 'B'
            self.score_b += 1

            # updata csv information
            self.Record('A', self.state, action, IsLaunch = is_launch, IsConstraint = False, violation = violation_reason_a)
            info = self.info('A', action, self.state, violation = True, constraint = False)

            return self.state, reward, info, done, False
        
        # Scenario 3. When A does not commit a foul
        else:
            # 1. First check if it exceeds the constraint
            if self.is_constraint:
                action, reward, isConstraint_a = self.badminton_constraint.validate_action(self.state, action)
            else:
                isConstraint_a = False

            # updata csv information
            self.Record('A', self.state, action, IsLaunch = is_launch, IsConstraint = isConstraint_a, violation = violation_reason_a)
            info = self.info('A', action, self.state, violation = True, constraint = False)

            # 2. Convert the action into B's state
            st = self.state[2]
            self.state = (action[0],
                        self.flip_coordinates(st),
                        self.flip_coordinates(action[3]), 
                        self.flip_coordinates(action[2])) 
            
            # 3. B agent generate the action 
            action_b = self.agent_b.action(self.state, self.info_dict, launch = False)
            
            # 4.After B completes the action, determine whether a foul has occurred
            
            # 4-1. When B commits a foul.
            if action_b is None or self.check_out_violation(action_b, self.state, is_launch = False) is not None:
                if action_b is None:
                    violation_reason_b = 'Miss hit'
                else:
                    violation_reason_b = self.check_out_violation(action_b, self.state, is_launch = False)

                # Next time A will serve, and update the score
                self.serving_player = 'A'
                self.score_a += 1

                # Update csv information 
                self.Record('B', self.state, action_b, IsLaunch = False, IsConstraint = False, violation = violation_reason_b)
                # Update reward, info, done
                reward = self.reward_function(Violation_reason = violation_reason_a, Mode = 'win', IsConstraint = isConstraint_a)
                info = self.info('B', action, self.state, violation = True, constraint = False)
                done = True

                return self.state, reward, info, done, False

            # 4-2. When B does not commit a foul
            else:
                # First check if it exceeds the constraint
                violation_reason_b = None
                if self.is_constraint:
                    action_b, _, isConstraint_b = self.badminton_constraint.validate_action(self.state, action_b)
                else:
                    isConstraint_b = False

                # Update csv information 
                self.Record('B', self.state, action_b, IsLaunch = False, IsConstraint = isConstraint_b, violation = violation_reason_b)
                # Update reward, info, done
                reward = self.reward_function(violation_reason_a, 'play', isConstraint_a)
                info = self.info('B', action, self.state, violation = False, constraint = isConstraint_b)
                done = False

                # The convert B's action into a state that A can accept and return it
                st = self.state[2]
                self.state = (action_b[0],
                            self.flip_coordinates(st), 
                            self.flip_coordinates(action_b[3]),
                            self.flip_coordinates(action_b[2]))

                return self.state, reward, info, done, False
    
    def info(self, player, action, state, violation, constraint):
        if self.round == 1:
            self.info_dict = {}
            if violation:
                self.info_dict['launch'] = False
            if constraint:
                self.info_dict['constraint'] = True
            
            self.info_dict['match'] = self.match
            self.info_dict['rally'] = self.rally
            self.info_dict['round'] = [self.round]
            if player == 'B':
                self.info_dict['score'] = [self.score_b, self.score_a]
                self.info_dict['player'] = [self.agent_b_name]
            else:
                self.info_dict['score'] = [self.score_a, self.score_b]
                self.info_dict['player'] = [self.agent_a_name]
            
            self.info_dict['env_score'] = [self.score_a, self.score_b]
            self.info_dict['state'] = [state]
            self.info_dict['action'] = [action]
            
        else:
            if violation:
                self.info_dict['launch'] = False
            if constraint:
                self.info_dict['constraint'] = True
            
            self.info_dict['match'] = self.match
            self.info_dict['rally'] = self.rally
            self.info_dict['round'].append(self.round)
            if player == 'B':
                self.info_dict['score'] = [self.score_b, self.score_a]
                self.info_dict['player'].append(self.agent_b_name)
            else:
                self.info_dict['score'] = [self.score_a, self.score_b]
                self.info_dict['player'].append(self.agent_a_name)
            self.info_dict['env_score'] = [self.score_a, self.score_b]
            self.info_dict['state'].append(state)
            self.info_dict['action'].append(action)
        
        return self.info_dict
    
    # ================== badminton scoring rules ================== #
    def check_winner(self):
        """Check if any player has won the game"""
        if self.score_a >= 21 and (self.score_a - self.score_b) >= 2:
            return 'A'
        elif self.score_b >= 21 and (self.score_b - self.score_a) >= 2:
            return 'B'
        elif self.score_a == 30:
            return 'A'
        elif self.score_b == 30:
            return 'B'
        return None

    def reset_for_next_game(self):
        """Reset the environment for the next game."""
        if self.check_winner():
            self.round = 1
            if self.is_match:
                self.match += 1
                self.rally = 1
                self.score_a = 0
                self.score_b = 0
                self.serving_player = 'A'  # A starts serving the new game
                self.current_side = 'A'
                self.state = None
    
    # ================== report saving / update match info ================== #
    def getPointer(self, score1, score2):
        if score1 != self.pre_score_a:
            self.pre_score_a = score1
            return self.agent_a_name # 'A'
        elif score2 != self.pre_score_b:
            self.pre_score_b = score2
            return self.agent_b_name #'B'
        else:
            return None

    def Record(self, player, state, action, IsLaunch, IsConstraint, violation):
        action_type_opponent, state_player, state_opponent, state_ball = state
        if action is None:
            action_type = 11
            action_hit = (0, 0)
            action_land = (0, 0)
            action_move = (0, 0)
        else:
            action_type, action_hit, action_land, action_move, _ = action

        if IsLaunch:
            self.round = 1

        winner = self.getPointer(self.score_a, self.score_b)
        row = pd.DataFrame([{'match': self.match,
                             'rally': self.rally,
                             'ball_round': self.round,
                             'player': player,
                             'serve': IsLaunch,
                             'roundscore_A': self.score_a,
                             'roundscore_B': self.score_b,
                             'player_location_x': state_player[0],
                             'player_location_y': state_player[1],
                             'opponent_location_x': state_opponent[0],
                             'opponent_location_y': state_opponent[1],
                             'receive_type_number': self.type_mapping.get(action_type_opponent),
                             'receive_type': action_type_opponent,
                             'type_number': self.type_mapping.get(action_type),
                             'type': action_type,
                             'player_hit_x': action_hit[0],
                             'player_hit_y': action_hit[1],
                             'landing_x': action_land[0],
                             'landing_y': action_land[1],
                             'player_movement_x': action_move[0],
                             'player_movement_y': action_move[1],
                             'getpoint_player': winner,
                             'constraint': IsConstraint,
                             'violation': violation
                            }])
        
        self.output = pd.concat([self.output, row])
        self.round += 1
        # Check if a match is satisfied or Check if a match is completed
        self.reset_for_next_game()

    def close(self):
        self.output.to_csv(self.filepath, index=False)