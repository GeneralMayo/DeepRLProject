#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

from hfo import *

import numpy as np
import tensorflow as tf

import keras
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


def init_actor_critic(init, new_relu):
    actor_input = Input(shape=(58,), name='actor_input') #beware of dtype
    actor_hidden1 = Dense(1024, kernel_initializer=init, activation=new_relu)(actor_input)
    actor_hidden2 = Dense(512, kernel_initializer=init, activation=new_relu)(actor_hidden1)
    actor_hidden3 = Dense(256, kernel_initializer=init, activation=new_relu)(actor_hidden2)
    actor_hidden4 = Dense(128, kernel_initializer=init, activation=new_relu)(actor_hidden3)

    # Output of the actor network. this is still part of the actor-critic network in our architecture and 
    # these two layers will not be accessed as model output but intermediate layer output
    # Note that they both take inputs from the actor_hidden4 layer.
    type_output = Dense(4, activation=new_relu, name="type_output")(actor_hidden4)
    param_output = Dense(6, activation='sigmoid', name="param_output")(actor_hidden4)

    x = keras.layers.concatenate([actor_input, type_output, param_output], axis=1)

    critic_hidden1 = Dense(1024, kernel_initializer=init, activation=new_relu)(x)
    critic_hidden2 = Dense(512, kernel_initializer=init, activation=new_relu)(critic_hidden1)
    critic_hidden3 = Dense(256, kernel_initializer=init, activation=new_relu)(critic_hidden2)
    critic_hidden4 = Dense(128, kernel_initializer=init, activation=new_relu)(critic_hidden3)

    critic_output = Dense(1, activation=new_relu, name="critic_output")(critic_hidden4)
    ac_model = Model(inputs=[actor_input], outputs=[critic_output])
    ac_model.compile(loss='mse', optimizer='adam')
    return ac_model


# def create_model(num_features, num_actions):
#   actor = Sequential()
#   actor.add(Dense(num_actions, input_shape=(num_features,), init='normal', use_bias=True, bias_initializer='normal'))

#   actorTarget = Sequential()
#   actorTarget.add(Dense(num_actions, input_shape=(num_features,), init='normal', use_bias=True, bias_initializer='normal'))

#   critic = Sequential()
#   critic.add(Dense(1, input_shape=(num_features+num_actions,), init='normal', use_bias=True, bias_initializer='normal'))

#   criticTarget = Sequential()
#   criticTarget.add(Dense(1, input_shape=(num_features+num_actions,), init='normal', use_bias=True, bias_initializer='normal'))

#   return [actor,actorTarget,critic,criticTarget]

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
  NUM_ITERATIONS = 500
  #TO_DO check max episode length
  MAX_EPISODE_LEN = 400
  
  #TO_DO check code
  BATCH_SIZE = 32
  #power and directon bounds respectively
  BOUNDS = [[0,100],[-180,180]]
  # Relu slope of negative region
  RELU_NEG_SLOPE = 0.01


  # Create the HFO Environment
  env = HFOEnvironment()
  # Connect to the server with the specified
  env.connectToServer(LOW_LEVEL_FEATURE_SET,
                      'bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)

  #create model
  #TO_DO
  # [actor,actorTarget,critic,criticTarget] = create_model(NUM_FEATURES, NUM_ACTIONS)
  init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
  new_relu = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE)
  actor_critic = init_actor_critic(init, new_relu)
  actor_critic_target = init_actor_critic(init, new_relu)
  policy = LinearDecayGreedyEpsilonPolicy(1,.1,10000,BOUNDS)
  #TO_DO (do we need a preprocessor? perhapse for bools of state)
  preprocessor = None
  memory = ReplayMemory(REPLAY_MEM_SIZE)
  agent = DQNAgent(actor_critic, actor_critic_target,preprocessor,memory,policy,GAMMA,SOFT_UPDATE_FREQ,SOFT_UPDATE_STEP,
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
