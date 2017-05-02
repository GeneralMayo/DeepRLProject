"""RL Policy classes.
"""
import numpy as np
from hfo import *


class LinearDecayGreedyEpsilonPolicy:
    """Policy with a parameter that decays linearly.

    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.

    Parameters
    ----------
    start_value: int, float
      The initial value of the parameter
    end_value: int, float
      The value of the policy at the end of the decay.
    num_steps: int
      The number of steps over which to decay the value.

    """

    def __init__(self, start_value, end_value, num_steps, bounds, params_squashed=True):  # noqa: D102
        assert num_steps > 0
        assert end_value <= start_value
        assert start_value >= 0
        assert start_value <= 1
        assert end_value >= 0
        assert end_value <= 1

        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        self.cur_steps = 0
        self.bounds = bounds
        self.params_squashed = params_squashed

    def get_unsquashed_params(self, action_values):
      action_values[4] = action_values[4]*100
      action_values[5] = action_values[5]*360-180
      action_values[6] = action_values[6]*360-180
      action_values[7] = action_values[7]*360-180
      action_values[8] = action_values[8]*100
      action_values[9] = action_values[9]*360-180
      return action_values

    def select_action(self, action_values, is_training):
        """Decay parameter and select action.

        Parameters
        ----------
        action_values: np.array
          The values from final layer of actor network.
          [[Dash, Turn, Tackle, Kick, p1dash, p2dash, p3turn, p4tackle, p5kick, p6kick]]
          NOTE: p1dash, p2dash, p3turn, p4tackle, p5kick, p6kick should be values from 0-1
        is_training: bool
          If true then parameter will be decayed.

        Returns
        -------
        selected_action: np.array
          ie:
          [Dash, p1dash, p2dash]
          [Turn, p3turn]
          [Tackle, p4tackle]
          [Kick, p5kick, p6kick]
        """

        if(self.params_squashed):
          action_values[0] = self.get_unsquashed_params(action_values[0])

        #calculate epsilon
        if(self.cur_steps >= self.num_steps):
            epsilon = self.end_value
        else:
            epsilon = self.start_value + (self.cur_steps/float(self.num_steps))*(self.end_value-self.start_value)

        self.cur_steps +=1

        if(epsilon>np.random.rand()):
          #choose random action type
          ACTION_TYPE = np.random.randint(0,4)

          #choose random parameters
          if(ACTION_TYPE == DASH or ACTION_TYPE == KICK):
            #random power/direction
            return [ACTION_TYPE, np.random.uniform(self.bounds[0][0],self.bounds[0][0]), np.random.uniform(self.bounds[1][0],self.bounds[1][0])]
          else:
            #random direction
            return [ACTION_TYPE, np.random.uniform(self.bounds[1][0],self.bounds[1][0])]
        else:
          #convert discreet action values to a list
          action_values_list = list(action_values[0,0:4])
          ACTION_TYPE = action_values_list.index(max(action_values_list))
          
          if(ACTION_TYPE == DASH):
            return [ACTION_TYPE,action_values[0,4],action_values[0,5]]
          elif(ACTION_TYPE == TURN):
            return [ACTION_TYPE,action_values[0,6]]
          elif(ACTION_TYPE == TACKLE):
            return [ACTION_TYPE,action_values[0,7]]
          elif(ACTION_TYPE == KICK):
            return [ACTION_TYPE,action_values[0,8],action_values[0,9]]
          else:
            raise NameError('Invalid ACTION_TYPE in policy')



    def reset(self):
        """Start the decay over at the start value."""
        self.cur_steps = 0
