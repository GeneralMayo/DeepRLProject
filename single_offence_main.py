#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

from hfo import *

import numpy as np
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

import shutil
import os
import sys
sys.path.insert(0, 'DeepRLProject/agent_utils')
from dqn import DQNAgent
from policy import LinearDecayGreedyEpsilonPolicy
from core import ReplayMemory
from actor_network import ActorNetwork
from critic_network import CriticNetwork

def get_param_bounds(BOUNDS):
  #assuming this param order --> [p1dash, p2dash, p3turn, p4tackle, p5kick, p6kick]
  lowerBounds = np.array([0, -180, -180, -180, 0, -180])
  upperBounds = np.array([100, 180, 180, 180, 100, 180])
  return [lowerBounds,upperBounds]

def main():
  debug = True
  if(debug):
    print("\n\nIN DEBUGGING MODE\n\n")
  else:
    print("\n\nNOT IN DEBUGGING MODE\n\n")

  use_old_init = False
  #make backup of tensorboard file then remove it
  
  #if(os.path.isdir('tensorboard_report')):
    #raise Exception('Back up tensorboard_report!')

  #set constants
  NUM_FEATURES = 58
  NUM_ACTION_TYPES = 4
  NUM_ACTION_PARAMS = 6
  EPSILON = .05
  #power and directon bounds respectively
  BOUNDS = [[0,100],[-180,180]]
  GAMMA = .99
  BATCH_SIZE = 32
  #Relu slope of negative region
  RELU_NEG_SLOPE = 0.01
  #max num of samples which can be stored in memory
  REPLAY_MEM_SIZE = 500000

  #debug constants
  if(debug):
    SOFT_UPDATE_FREQ = 1
    SOFT_UPDATE_STEP = .001
    MAX_EPISODE_LEN = 400
    MEMORY_THRESHOLD = 100
    NUM_ITERATIONS = 3000
    EPISODES_PER_EVAL = 2
    EVAL_FREQ = 500
    SAVE_FREQ = 500
    LEARNING_RATE = .001
    POLICY_EXPLORATION_STEPS = 100
  #actual constants
  else:
    SOFT_UPDATE_FREQ = 1
    SOFT_UPDATE_STEP = .001
    MAX_EPISODE_LEN = 400
    MEMORY_THRESHOLD = 1000
    NUM_ITERATIONS = 750000
    EPISODES_PER_EVAL = 20
    EVAL_FREQ = 5000
    SAVE_FREQ = 100000
    LEARNING_RATE = .001
    POLICY_EXPLORATION_STEPS = 10000

  # Create the HFO Environment
  env = HFOEnvironment()
  # Connect to the server with the specified
  env.connectToServer(LOW_LEVEL_FEATURE_SET,
                      'bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)

  #create models
  actor = ActorNetwork(sess, NUM_FEATURES, NUM_ACTION_TYPES, NUM_ACTION_PARAMS, RELU_NEG_SLOPE, LEARNING_RATE)
  critic = CriticNetwork(sess, NUM_FEATURES, NUM_ACTION_TYPES, NUM_ACTION_PARAMS, get_param_bounds(BOUNDS), BATCH_SIZE,
              RELU_NEG_SLOPE,LEARNING_RATE) 

  init_op = tf.global_variables_initializer()
  sess.run(init_op)

  #agent policy
  policy = LinearDecayGreedyEpsilonPolicy(1, .1, POLICY_EXPLORATION_STEPS, BOUNDS)
  #replay mem
  memory = ReplayMemory(REPLAY_MEM_SIZE)
  #DQ Actor/ Critic Agent
  agent = DQNAgent(actor, critic, memory, policy, GAMMA, SOFT_UPDATE_FREQ, SOFT_UPDATE_STEP,
        BATCH_SIZE, MEMORY_THRESHOLD, NUM_ACTION_TYPES, NUM_ACTION_PARAMS, NUM_FEATURES, EVAL_FREQ, EPISODES_PER_EVAL, SAVE_FREQ)
  
  #save actor to use later
  actor_model_str="actor_model.ymal"
  actor_yaml = actor.online_network.to_yaml()
  with open(actor_model_str, "w") as yaml_file:
    yaml_file.write(actor_yaml)

  #save critic to use later
  critic_model_str="critic_model.ymal"
  critic_yaml = critic.online_network.to_yaml()
  with open(critic_model_str, "w") as yaml_file:
    yaml_file.write(critic_yaml)

  #train agent
  #try:
  agent.fit(env, NUM_ITERATIONS, MAX_EPISODE_LEN)
  """
    print("FINISHED TRAINING")
  except(Exception):
    print("EMERGENCY_SAVE")
    agent.save_weights(6666)
    agent.save_replay_memory(6666)
  """

if __name__ == '__main__':
  main()
