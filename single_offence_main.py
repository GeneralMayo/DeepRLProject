#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

from hfo import *

import numpy as np
import tensorflow as tf

from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute,Reshape,merge)
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras import losses
from keras import backend

import sys
sys.path.insert(0, 'DeepRLProject/agent_utils')
from dqn import DQNAgent
from policy import LinearDecayGreedyEpsilonPolicy
from core import ReplayMemory


def create_model(num_features, num_actions):
  actor = Sequential()
  actor.add(Dense(num_actions, input_shape=(num_features,), init='normal', use_bias=True, bias_initializer='normal'))

  actorTarget = Sequential()
  actorTarget.add(Dense(num_actions, input_shape=(num_features,), init='normal', use_bias=True, bias_initializer='normal'))

  critic = Sequential()
  critic.add(Dense(1, input_shape=(num_features+num_actions,), init='normal', use_bias=True, bias_initializer='normal'))

  criticTarget = Sequential()
  criticTarget.add(Dense(1, input_shape=(num_features+num_actions,), init='normal', use_bias=True, bias_initializer='normal'))


  return [actor,actorTarget,critic,criticTarget]

def main():
  #set constants
  NUM_FEATURES = 58
  NUM_ACTIONS = 10
  EPSILON = .05
  REPLAY_MEM_SIZE = 500000
  MEMORY_THRESHOLD = 32
  GAMMA = .99
  SOFT_UPDATE_FREQ = 1
  SOFT_UPDATE_STEP = .001
  #TO_DO check max num iterations
  NUM_ITERATIONS = 1000
  #TO_DO check max episode length
  MAX_EPISODE_LEN = 400
  
  #TO_DO check code
  BATCH_SIZE = 32
  #power and directon bounds respectively
  BOUNDS = [[0,100],[-180,180]]


  # Create the HFO Environment
  env = HFOEnvironment()
  # Connect to the server with the specified
  env.connectToServer(LOW_LEVEL_FEATURE_SET,
                      'bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)

  #create model
  #TO_DO
  [actor,actorTarget,critic,criticTarget] = create_model(NUM_FEATURES, NUM_ACTIONS)
  policy = LinearDecayGreedyEpsilonPolicy(1,.1,10000,BOUNDS)
  #TO_DO (do we need a preprocessor? perhapse for bools of state)
  preprocessor = None
  memory = ReplayMemory(REPLAY_MEM_SIZE)
  agent = DQNAgent(actor,actorTarget,critic,criticTarget,preprocessor,memory,policy,GAMMA,SOFT_UPDATE_FREQ,SOFT_UPDATE_STEP,
        BATCH_SIZE,MEMORY_THRESHOLD,NUM_ACTIONS,NUM_FEATURES)

  #compile agent
  #TO_DO figure out learning rate
  adam = Adam(lr=0.0001)
  #TO_DO: is this the loss which should be used? or should it be MSE
  loss = 'mean_squared_error'
  agent.compile(adam,loss)

  #train agent
  agent.fit(env, NUM_ITERATIONS, MAX_EPISODE_LEN)

  print("FINISHED TRAINING")

if __name__ == '__main__':
  main()
