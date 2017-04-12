#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import itertools
from hfo import *

def create_model(INPUT_SHAPE, NUM_ACTIONS):
  pass

def main():
  #set constants
  INPUT_SHAPE = 58
  NUM_ACTIONS = 7
  EPSILON = .05
  REPLAY_MEM_SIZE = 500000
  MEMORY_THRESHOLD = 1000
  GAMMA = .99
  SOFT_UPDATE_FREQ = 1
  SOFT_UPDATE_STEP = .001
  NUM_ITERATIONS = 
  
  #TO_DO check code
  BATCH_SIZE = 32


  # Create the HFO Environment
  hfo = HFOEnvironment()
  # Connect to the server with the specified
  hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                      'bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)

  
  #create model
  #TO_DO
  model = create_model(INPUT_SHAPE, NUM_ACTIONS)
  #TO_DO include this function
  policy = GreedyEpsilonPolicy(EPSILON)
  #TO_DO (do we need a preprocessor? perhapse for bools of state)
  preprocessor = Preprocessor()
  #TO_DO
  memory = ReplayMemory(REPLAY_MEM_SIZE)
  #TO_DO
  agent = DQNAgent(model,preprocessor,memory,policy,GAMMA,SOFT_UPDATE_FREQ,SOFT_UPDATE_STEP,
        BATCH_SIZE,MEMORY_THRESHOLD,NUM_ACTIONS)

  #compile agent
  adam = Adam(lr=0.0001)
  #TO_DO: is this the loss which should be used? or should it be MSE
  loss = mean_huber_loss
  agent.compile(adam,loss)

  #train agent
  agent.fit(env, NUM_ITERATIONS, MAX_EPISODE_LEN)

  print("FINISHED TRAINING")

  for episode in itertools.count():
    status = IN_GAME
    while status == IN_GAME:
      # Grab the state features from the environment
      features = hfo.getState()
      print(len(features))
      print(type(features))
      print(np.shape(features))
      print(features)
      input()

      # Take an action and get the current game status
      hfo.act(DASH, 20.0, 0.)
      # Advance the environment and get the game status
      status = hfo.step()
    # Check the outcome of the episode
    print('Episode %d ended with %s'%(episode, hfo.statusToString(status)))
    # Quit if the server goes down
    if status == SERVER_DOWN:
      hfo.act(QUIT)
      break

if __name__ == '__main__':
  main()
