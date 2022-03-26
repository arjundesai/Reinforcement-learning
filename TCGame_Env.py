from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

   def __init__(self, winning_sum):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix
        
        self.winning_sum = winning_sum

        self.reset()
        
   def get_env_action(self, curr_state):
        env_actions = list(self.action_space(curr_state)[1])
        selected_action = random.choice(env_actions)
       
        return selected_action
        
        
   def reward(self, state_result, is_agent_action):
        if state_result == 'Win':
            return 10 * (1 if is_agent_action else -1)
        elif state_result == 'Tie':
            return 0
        else:
            return -1
        

   def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        
        grid = np.array(curr_state).reshape((3,3))
        
        grid_sum = list(np.sum(grid, axis = 1))
        grid_sum.extend(list(np.sum(grid, axis = 0)))
        grid_sum.append(np.trace(grid))
        grid_sum.append(np.trace(np.fliplr(grid)))
        
        output = [sum_value for sum_value in grid_sum if not np.isnan(sum_value) and sum_value == self.winning_sum]
        return (len(output) >= 1)
    

   def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


   def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


   def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


   def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)


   def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        next_state = curr_state.copy()
        next_state[curr_action[0]] = curr_action[1]
        return next_state


   def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        next_state = self.state_transition(curr_state, curr_action)
        state_result = self.is_terminal(next_state)
        
        if state_result[0]:
            reward = self.reward(state_result[1], True)
        else:
            env_action = self.get_env_action(next_state)
            next_state = self.state_transition(next_state, env_action)
            state_result = self.is_terminal(next_state)
            reward = self.reward(state_result[1], False)
            
        return (next_state, reward, state_result[0])


   def reset(self):
        return self.state
