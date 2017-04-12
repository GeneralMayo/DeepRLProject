import numpy as np
#import matplotlib.pyplot as plt
import time
import random

"""Main DQN agent."""

class DQNAgent:
    """Class implementing DQN / Liniear QN.

                 
    Parameters
    ----------
    q_network: keras.models.Model
      Online Q-network model.
    target_q_network: keras.models.Model
      Q-network used for target fixing method.
    preprocessor: deeprl_hw2.core.Preprocessor
      See the associated classes for more details.
    history: deeprl_hw2.preprocessor.HistoryPreprocessor
      See the associated classes for more details.
    memory: deeprl_hw2.core.Memory
      Replay memory.
    policy: deeprl_hw2.policy.Policy
      Policy followed by agent during training. See the associated 
      classes for more details.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    replay_start_size: int
      Total number of samples burned in before training starts
    num_actions: int
      Total number of actions agent can choose.
    network_type: str
      Type of DQN/Linear QN
    reward_samp: int
      Number of iterations between each evaluation session
    held_out_states: deeprl_hw2.core.Memory
      Memory to store held_out_states
    held_out_states_size: int
      Number of states held out as described in https://arxiv.org/pdf/1312.5602.pdf
    train_freq: int
      Number of iterations between each online network update
    """
    def __init__(self,
                 q_network,
                 target_q_network,
                 preprocessor,
                 history,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 batch_size,
                 replay_start_size,
                 num_actions,
                 network_type,
                 reward_samp,
                 held_out_states,
                 held_out_states_size,
                 train_freq):
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.preprocessor = preprocessor
        self.history=history
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.replay_start_size=replay_start_size
        self.num_actions = num_actions
        self.network_type = network_type
        self.reward_samp = reward_samp
        self.held_out_states = held_out_states
        self.held_out_states_size = held_out_states_size
        self.train_freq=train_freq

        #This is the pointer to the function called when a particular type of QN
        #is traied on a batch. (Determined when "fit" function is called.)
        self.update_policy = None

    def compile(self, optimizer, loss_func):
        """
          Compile online network (and target network if target fixing/ double Q-learning
            is being utilized) with provided optimizer/loss functions.
        """

        self.q_network.compile(loss=loss_func,optimizer=optimizer)

        #compile target network if this strategy is being used
        if(self.target_q_network != None):
            self.target_q_network.compile(loss=loss_func,optimizer=optimizer)

    def calc_q_values(self, state):
        """
        Given a preprocessed state (or batch of states) calculate the Q-values.

        Parameters
        ----------
        state: numpy.array
          size is (32x4x84x84) for batch and (1x4x84x84) for single sample.
        
        Return
        ------
        Q-values for the state(s): numpy.array
        """
        return self.q_network.predict(state)

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.
        
        Parameters
        ----------
        state: list of numpy.array
          size of list = 4
          size of np array = 84x84
          ie: Each elem of the list is a frame.

        Returns
        --------
        selected action: int
        """

        #state represented as a list to stat represented as an np array
        state = self.preprocessor.process_state_for_network(state)
        #get q-values of each action for given state
        q_vals = self.calc_q_values(state)
        #select action based on given q-values and agent's policy
        return self.policy.select_action(q_vals, self.num_actions)


    def getInputAndNextStates(self, minibatch):
        """
        Extracts current and next states from a minibatch and stores them 
        in a form which can be processed by the Q-network in parallel during
        forward propegation steps.

        This is a common task preformed in "update_policy" functions.

        Parameters
        ----------
        minibatch: list of deeprl_hw2.core.Sample objects
          size of list = batch_size

        Returns
        --------
        A list of 2 np arrays.
        1st np array = s_t's of minibatch
        2nd np array = s_t1's of minibatch
        """

        #init np arrays to hold s_t's and s_t1's of minibatch
        inputStates = np.zeros((self.batch_size, 4, 84,84))
        nextStates = np.zeros((self.batch_size, 4, 84,84))
        
        for sampleIdx in range(self.batch_size):
            #Decouple current and next state which are stored in the minibatch
            #as [f1,f2,f3,f4,f5]
            #After decoupling states will be...
            #s_t = [f1,f2,f3,f4]
            #s_t1 = [f2,f3,f4,f5]
            states=(minibatch[sampleIdx].states)
            s_t=list()
            s_t1=list()
            for i in range(4):
                s_t.append(states[i])
                s_t1.append(states[i+1])
            
            #convert list of frames --> np array of frames
            s_t = self.preprocessor.process_state_for_network(s_t)
            inputStates[sampleIdx] = s_t
            s_t1 = self.preprocessor.process_state_for_network(s_t1)
            nextStates[sampleIdx] = s_t1

        return [inputStates, nextStates]

    def Linear_update_policy(self,s_t, q_t, a_t, r_t, s_t1, is_terminal):
        """
        Compute weight updates for linear Q-network.

        Parameters
        ----------
        s_t: np.array (1x4x84x84)
          Current state.
        q_t: np.array (num-actions)
          q-values estimated by online network
        a_t: int
          Action taken by online network given s_t
        r_t: int
          Reward gained by online network by taking action a_t from s_t
        s_t1: np.array (1x4x84x84)
          Next state.
        is_terminal: bool
          True if s_t1 is a terminal state.

        Returns
        --------
        loss: float
          batch training loss
        """
        
        #Note: "batch size" is 1 for this network

        #set all q-values of target to q-values generated by online network
        #such that the losses generated by q-values associated with actions not
        #taken by the online network are 0
        target = q_t

        #modify q-value of the target associated with the action taken by the
        #online network to be the target q-value described by q-learning algorithm
        if(is_terminal):
            target[0][a_t] = r_t
        else:
            q_t1 = self.calc_q_values(s_t1)
            target[0][a_t] = self.gamma*max(q_t1[0]) + r_t

        #update weights/ return loss
        return self.q_network.train_on_batch(s_t,target)

    def DQN_and_LinearERTF_update_policy(self):
        """
        Compute weight updates for either...
        1) DQN or 
        2) Linear Q-network with experience replay & target fixing

        Returns
        --------
        loss: float
          batch training loss
        """
        #get minibatch
        minibatch = self.memory.sample(self.batch_size)
        minibatch = self.preprocessor.process_batch(minibatch)

        #init state inputs + state targets
        [inputStates,nextStates] = self.getInputAndNextStates(minibatch)

        #Set all q-values of target to q-values generated by online network
        #such that the losses generated by q-values associated with actions not
        #taken by the online network are 0.
        targets = self.calc_q_values(inputStates)
        #Forward propegation step used in Q-learning algorithm
        q_t1_All = self.target_q_network.predict(nextStates)
        
        #modify q-value of the target associated with the action taken by the
        #online network to be the target q-value described by q-learning algorithm
        for sampleIdx in range(self.batch_size):
            a_t=minibatch[sampleIdx].a_t 
            r_t=minibatch[sampleIdx].r_t
            is_terminal=minibatch[sampleIdx].is_terminal

            if(is_terminal):
                targets[sampleIdx][a_t] = r_t
            else:
                targets[sampleIdx][a_t] = self.gamma*max(q_t1_All[sampleIdx]) +r_t

        #update weights
        loss = self.q_network.train_on_batch(inputStates,targets)
        
        return loss

    def DoubleLinear_update_policy(self):
        """
        Compute weight updates for Double Linear Q-network.
        
        Returns
        --------
        loss: float
          batch training loss
        """
        #choose online/ target network (coin flip)
        if(random.randint(0,1)==1):
            temp = self.q_network
            self.q_network = self.target_q_network
            self.target_q_network = temp

        #follow DDQN policy
        return self.DDQN_update_policy()

    def DDQN_update_policy(self):
        """
        Compute weight updates for DDQN.
        
        Returns
        --------
        loss: float
          batch training loss
        """

        #get minibatch
        minibatch = self.memory.sample(self.batch_size)
        minibatch = self.preprocessor.process_batch(minibatch)

        #init state inputs + state targets
        [inputStates,nextStates] = self.getInputAndNextStates(minibatch)

        #Set all q-values of target to q-values generated by online network
        #such that the losses generated by q-values associated with actions not
        #taken by the online network are 0.
        targets = self.calc_q_values(inputStates)
        #Forward propegation steps used in Double Q-learning algorithm
        q_t1_online_All = self.calc_q_values(nextStates)
        q_t1_target_All = self.target_q_network.predict(nextStates)

        #modify q-value of the target associated with the action taken by the
        #online network to be the target q-value described by q-learning algorithm
        for sampleIdx in range(self.batch_size):
            a_t=minibatch[sampleIdx].a_t 
            r_t=minibatch[sampleIdx].r_t
            is_terminal=minibatch[sampleIdx].is_terminal

            if(is_terminal):
                targets[sampleIdx][a_t] = r_t
            else:
                #best action according to online network will be the action which target network chooses
                a_t1_online = np.argmax(q_t1_online_All[sampleIdx])
                targets[sampleIdx][a_t] = self.gamma*q_t1_target_All[sampleIdx][a_t1_online]+r_t


        #update weights
        loss = self.q_network.train_on_batch(inputStates,targets)

        return loss

    
    def save_weights_on_interval(self, curiter, totaliter):
        """
        Saves model and weights at different stages of training.

        Parameters
        ----------
        curiter: int
          current iteration in training
        totaliter: int
          total number of iterations agent will train for

        Saves
        --------
        json file: 
          representing current state of agent
        weights file:
          representing current weights of agent
        """
        
        #agent noticably improves every 500000 iterations 
        if (curiter % 500000==0):
            stage=int(curiter/500000)
            modelstr=self.network_type+"model"+str(stage)+".json"
            weightstr=self.network_type+"model"+str(stage)+".h5"
            model_json = self.q_network.to_json()
            with open(modelstr, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.q_network.save_weights(weightstr)


    def populate_replay_memory_and_held_out_states(self, env):
        """
        Populates replay memory and held out states memory with samples
        selected using a random policy.

        Parameters
        ----------
        env: gym.Env
          Atari environment being simulated.
        """
        print("Populating replay mem ...")
        #initial state for replay memory filling
        self.history.process_state_for_network(env.reset())
        s_t=self.history.frames

        # populate replay memory
        for iter in range(self.replay_start_size):
            if(iter%10000 == 0):
                print("Replay Mem Iter: "+str(iter))
            
            # select random action
            a_t = env.action_space.sample()
            
            # get next state, reward, is terminal
            (image, r_t, is_terminal, info) = env.step(a_t)
            r_t=self.preprocessor.process_reward(r_t)
            self.history.process_state_for_network(image)
            s_t1=self.history.frames

            #store held out states to be used for Q-value evaluation
            if iter<self.held_out_states_size:
                self.held_out_states.append(self.preprocessor.process_state_for_memory(s_t), a_t,
                               r_t, self.preprocessor.process_state_for_memory(s_t1), is_terminal)
            
            # store sample in memory
            if(self.network_type != "Linear"):
                self.memory.append(self.preprocessor.process_state_for_memory(s_t), a_t,
                        r_t, self.preprocessor.process_state_for_memory(s_t1), is_terminal)
            
            #update new state
            if (is_terminal):
                self.history.reset()
                self.history.process_state_for_network(env.reset())
                s_t = self.history.frames
            else:
                s_t = s_t1

        print("Done populating replay mem ...")

    def update_target(self,iteration):
        """
        Used to update target q-network network with online network weights every
        "target_update_freq" iterations.

        Parameters
        __________
        iteration: int
          current iteration in training

        """
        if(iteration % self.target_update_freq == 0 and
            self.network_type != "Linear" and       #has no target network
            self.network_type != "DoubleLinear"):   #target update replaced with "coin-flip"

            print ("Updating target Q network")
            self.target_q_network.set_weights(self.q_network.get_weights())
        

    def set_update_policy_function(self):
        """
        Select function to update this particular agent's policy.
        """

        if(self.network_type == "Linear"):
            self.update_policy = self.Linear_update_policy
        elif(self.network_type == "LinearERTF" or self.network_type == "DQN"):
            self.update_policy = self.DQN_and_LinearERTF_update_policy
        elif(self.network_type == "DoubleLinear"):
            self.update_policy = self.DoubleLinear_update_policy
        elif(self.network_type == "DDQN"):
            self.update_policy = self.DDQN_update_policy
        elif(self.network_type == "Duling"):
            #Note: Duling DQN can be trained with either normal DQN or DDQN update policy
            self.update_policy = self.DQN_and_LinearERTF_update_policy
            #self.update_policy = self.DDQN_update_policy
        else:
            raise ValueError("Invalid network type.")

    def fit(self, env, num_iterations, max_episode_length,num_episodes=20):
        """Fit DQN/Linear QN model to the provided environment.

        Parameters
        ----------
        env: gym.Env
          Environment to be fitted.
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        
        #populate replay memory (if network has one)
        self.populate_replay_memory_and_held_out_states(env)

        #set update policy function 
        #(update policy depends on type of network being trained and Q-learning
        #algorithm being used)
        self.set_update_policy_function()

        #get initial state
        self.history.process_state_for_network(env.reset())
        s_t = self.history.frames

        #init metric vectors (to eventually plot)
        allLoss=np.zeros(num_iterations)
        rewards=np.zeros(int(np.ceil(num_iterations/self.reward_samp)))
        avg_qvals_iter=np.zeros(int(np.ceil(num_iterations/self.reward_samp)))

        #iterate through environment samples
        for iteration in range(num_iterations):
            
            #check if target needs to be updated (all network types handeled appropriately)
            self.update_target(iteration)

            # this function saves weights 0/3, 1/3, 2/3, and 3/3 of the way through training
            self.save_weights_on_interval(iteration, num_iterations)

            #select action
            if(self.network_type == "Linear"):
                #convert list of frames --> np array of frames
                s_t = self.preprocessor.process_state_for_network(s_t)
                #get q-vals
                q_t = self.calc_q_values(s_t)
                #select action
                a_t = self.policy.select_action(q_t, self.num_actions)
            else:
                a_t = self.select_action(s_t)
         
            #get next state, reward, is terminal
            (image, r_t, is_terminal, info) = env.step(a_t)
            r_t=self.preprocessor.process_reward(r_t)
            self.history.process_state_for_network(image)
            s_t1 = self.history.frames

            #store sample in memory (convert states to uint8 first)
            if(self.network_type != "Linear"):
                self.memory.append(self.preprocessor.process_state_for_memory(s_t), a_t,
                               r_t, self.preprocessor.process_state_for_memory(s_t1),is_terminal)

            #update policy
            if (iteration%self.train_freq==0):
                if(self.network_type != "Linear"):
                    loss = self.update_policy()
                else:
                    #convert list of frames --> np array of frames
                    s_t1 = self.preprocessor.process_state_for_network(s_t1)
                    loss = self.update_policy(q_t,s_t,a_t,r_t,s_t1,is_terminal)

                allLoss[iteration] = loss

            #data logging
            if (iteration==0):
                print ("Training Starts")
                with open('testlog.txt',"a") as f:
                    f.write("Training Starts\n")

            #evaluation of policy
            if (iteration % self.reward_samp == 0):               
                print("Start Evaluation\n")

                with open('testlog.txt', "a") as f:
                    f.write("Start Evaluation\n")

                #get evaluation metrics
                cum_reward, avg_qvals, avg_max_qval, avg_min_qval = self.evaluate(env,num_episodes,max_episode_length)
                
                #store evaluation metrics
                rewards[int(iteration / self.reward_samp)] = cum_reward
                avg_qvals_iter[int(iteration / self.reward_samp)] = avg_qvals
                prtscn ="At iteration : " + str(iteration) + " , Average Reward = " + str(cum_reward)+ " , Average Q value = " +str(avg_qvals)+" , Min Q value = "+str(avg_min_qval)+" , Max Q value = "+str(avg_max_qval)+" , Loss = " +str(loss)+"\n"
                print (prtscn)
                with open('testlog.txt', "a") as f:
                    f.write(prtscn)
            
            #occasionally save a copy of evaluation metrics
            if (iteration % 500000 == 0):
                np.save(self.network_type+"loss_linear_MR_TF", allLoss)
                np.save(self.network_type+"reward_linear_MR_TF", rewards)
                np.save(self.network_type+"avg_qvals_iter", avg_qvals_iter)

            #update new state
            if (is_terminal):
                self.history.reset()
                self.history.process_state_for_network(env.reset())
                s_t = self.history.frames
            else:
                s_t = s_t1

        print("DONE TRAINING")
        

        """
        fig = plt.figure()
        plt.plot(allLoss)
        plt.ylabel('Loss function')
        fig.savefig('Loss.png')
        plt.clf()
        plt.plot(rewards)
        plt.ylabel('Average Reward')
        fig.savefig('reward.png')
        plt.clf()
        plt.plot(avg_qvals_iter)
        plt.ylabel('Average Q value')
        fig.savefig('q_value.png')
        """

    def evaluate(self, env, num_episodes, max_episode_length):
        """
        Evaluate policy.
        """

        cumulative_reward = 0
        actions = np.zeros(env.action_space.n)
        no_op_max=30

        for episodes in range(num_episodes):
            steps = 0
            q_vals_eval=np.zeros(self.held_out_states_size)
            held_out=list(self.held_out_states.M) # convert the deque into a list of samples
            for i in range(self.held_out_states_size):                
                state = held_out[i].states[0:4]  #get the initial state from the sample
                state = self.preprocessor.process_state_for_network(state) #conversion into float
                q_vals = self.calc_q_values(state)              
                q_vals_eval[i]=q_vals_eval[i]+max(q_vals[0])  #take the max over actions and add to the current state
                
            # get initial state
            self.history.reset()
            self.history.process_state_for_network(env.reset())
            state = self.history.frames
            for i in range(no_op_max):
                (next_state, reward, is_terminal, info) = env.step(0)
                self.history.process_state_for_network(next_state)
                next_state = self.history.frames
                actions[0] += 1
                steps = steps + 1
                if is_terminal:
                    state=env.reset()
                else:
                    state=next_state
                  
            
            while steps < max_episode_length:
                state = self.preprocessor.process_state_for_network(state)
                q_vals = self.calc_q_values(state)
                action = np.argmax(q_vals[0])
                actions[action] += 1
                (next_image, reward, is_terminal, info) = env.step(action)
                cumulative_reward = cumulative_reward + reward
                self.history.process_state_for_network(next_image)
                next_state = self.history.frames
                state = next_state
                steps = steps + 1
                if is_terminal:
                    break

        avg_reward = cumulative_reward / num_episodes
        avg_qval=np.mean(q_vals_eval)/num_episodes
        avg_max_qval=max(q_vals_eval)/num_episodes
        avg_min_qval=min(q_vals_eval)/num_episodes
        
        return avg_reward, avg_qval,avg_max_qval,avg_min_qval
