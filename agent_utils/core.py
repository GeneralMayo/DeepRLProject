"""Core classes."""
import numpy as np
import gym
import cv2
import random
from collections import deque



class Sample:
    #TO_DO
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Parameters
    ----------
    s_t: np.array of features (58,)
      Current state.
    a_t: np.array of action values  [Dash, Turn, Tackle, Kick, p1dash, p2dash, p3turn, p4tackle, p5kick, p6kick]
      Action taken by online network given s_t
    r_t: float
      Reward gained by online network by taking action a_t from s_t
    s_t1: np.array (58,)
      Next state.
    is_terminal: bool
      True if s_t1 is a terminal state.
    """
    def __init__(self,s_t,a_t,r_t,s_t1,is_terminal):        
        pass
        
class Preprocessor:
    """
    Preprocessor base class.
    """
    def __init__(self):
        pass

    def process_state_for_network(self, state):
        pass
        
    def process_state_for_memory(self, state):
        pass

    def process_batch(self, samples):
        pass

    def process_reward(self, reward):
        pass

class ReplayMemory:
    """
    Class implementing a replay memory which only stores a specified number
    of most recent samples.


    Parameters
    ----------
    max_size: int
        maximum amount of samples which can be stored in this replay memory
    window_length: int
        number of frames stored in each state

    """
    def __init__(self, max_size, window_length):
        #Note: Once a bounded length deque is full, when new items are added, a 
        #corresponding number of items are discarded from the opposite end.
        self.M = deque(maxlen=max_size)
        self.max_size = max_size
        
    def append(self, state, action, reward, next_state, is_terminal):
        self.M.append(Sample(state,action,reward,next_state,is_terminal))


    def sample(self, batch_size):
        """
        Samples a batch from the replay memory.

        Parameters
        ----------
        batch_size: int
          Size of batch to be sampled from the replay memory

        Returns
        -------
        minibatch: list of deeprl_hw2.core.Sample objects
          size of list = batch_size

        """
        return random.sample(list(self.M), batch_size)

    def clear(self):
        self.M.clear()
