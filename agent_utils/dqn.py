import numpy as np
#import matplotlib.pyplot as plt
import time
import random

"""Main DQN agent."""

class DQNAgent:
    """Class implementing DQN / Liniear QN.
    """
    DQNAgent(model,preprocessor,memory,policy,GAMMA,SOFT_UPDATE_FREQ,SOFT_UPDATE_STEP,
        BATCH_SIZE,MEMORY_THRESHOLD,NUM_ACTIONS)    


    def __init__(self,
                 actor,
                 actor_target,
                 critic,
                 critic_target,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 soft_update_freq,
                 soft_update_step,
                 batch_size,
                 memory_threshold,
                 num_actions,
                ):

        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.soft_update_freq = soft_update_freq
        self.soft_update_step = soft_update_step
        self.batch_size = batch_size
        self.memory_threshold = memory_threshold
        self.num_actions = num_actions

    #TO_DO
    def compile(self, optimizer, loss_func):
        """
          Compile online network (and target network if target fixing/ double Q-learning
            is being utilized) with provided optimizer/loss functions.
        """
        pass 

    #TO_DO
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
        pass

    
    #TO_DO probably need to rewrite this
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

    def advance_environment(self, env):
        status = env.step()

        is_terminal = False

        # Quit if the server goes down
        if status == SERVER_DOWN:
            #TO_DO: save all models + replay memory
            env.act(QUIT)
        elif status != IN_GAME:
            is_terminal = True
        

        return [status, is_terminal]

    def load_action(self, env, action_vector):
        #get chosen action
        action_type_and_params = self.policy.select_action(action_vector, True)
        if(len(action_type_and_params) == 3):
            env.act(action_type_and_params[0], action_type_and_params[1], action_type_and_params[2])
        else:
            env.act(action_type_and_params[0], action_type_and_params[1])

    def get_reward(self, s_t, s_t1, action_vector):
        """
        Parameters
        ----------
        s_t: np.array (58,)
            The current as given by hfo.getState()
        s_t1: np.array (58,)
            The next state as given by hfo.getState()
        action_vector: np.array(,10)
            [Dash, Turn, Tackle, Kick, p1dash, p2dash, p3turn, p4tackle, p5kick, p6kick]

        Returns
        -------
        reward: float
            The reward gained from transitioning between s_t and s_t1
        """

        pass

    def fit(self, env, num_iterations, max_episode_length, num_episodes=20):
        """Fit DQN/Linear QN model to the provided environment.

        Parameters
        ----------
        env: HFOEnvironment
          Environment to be fitted.
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        
        #populate replay memory (if network has one)
        #TO_DO implement this correctly
        self.populate_replay_memory_and_held_out_states(env)

        #init metric vectors (to eventually plot)
        #TO_DO implement logging
        #loss_log=np.zeros(num_iterations)
        #reward_log=np.zeros(int(np.ceil(num_iterations/self.reward_samp)))

        #iterate through environment samples
        cur_iteration = 0
        for episode in itertools.count():
            #indicate game has started
            status = IN_GAME
            #get initial state
            s_t = hfo.getState()

            while status == IN_GAME:
                if(cur_iteration == num_iterations):
                    #TO_DO save everything
                    #possibly save replay memory
                    return

                #TO_DO: update network
                loss = self.update_network()

            
                #TO_DO: get action vector
                action_vector = self.model. (get action vector)

                #loads action ie: env.act(DASH, 20.0, 0.)
                self.load_action(env, action_vector)

                #advance the environment and get the game status
                [status, is_terminal] = self.advance_environment(env)

                #get new state
                s_t1 = env.getState()

                #get reward
                r_t = self.get_reward(s_t, s_t1, action_vector)

                #store sample
                self.memory.append(s_t, action_vector, r_t, s_t1, is_terminal)

                #set new current state
                s_t = s_t1

            # Check the outcome of the episode
            print('Episode %d ended with %s'%(episode, hfo.statusToString(status)))
            # Quit if the server goes down
            if status == SERVER_DOWN:
                hfo.act(QUIT)
                break


        #TO_DO: save all models... + log files..


    #TO_DO: rewrite evaluate
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
