"""RL Policy classes.
"""
import numpy as np
import attr

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

    def __init__(self, start_value, end_value, num_steps):  # noqa: D102
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

    def select_action(self, action_values, num_actions, is_training):
        """Decay parameter and select action.

        Parameters
        ----------
        action_values: np.array
          The values from final layer of actor network.
          [Dash, Turn, Tackle, Kick, p1dash, p2dash, p3turn, p4tackle, p5kick, p6kick]
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
        #calculate epsilon
        if(self.cur_steps >= self.num_steps):
            epsilon = self.end_value
        else:
            epsilon = self.start_value + (self.cur_steps/float(self.num_steps))*(self.end_value-self.start_value)

        self.cur_steps +=1

        #TO_DO implement action selection as in paper



    def reset(self):
        """Start the decay over at the start value."""
        self.cur_steps = 0
