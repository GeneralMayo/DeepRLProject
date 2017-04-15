from hfo import *
import numpy as np
import time
import random
import itertools
import math
from math import sqrt
from math import cos
from keras import backend as K
import keras
import tensorflow as tf

"""Main DQN agent."""

class DQNAgent:
    """Class implementing DQN / Liniear QN.
    """ 

    def __init__(self,
                 actor_critic,
                 actor_critic_target,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 soft_update_freq,
                 soft_update_step,
                 batch_size,
                 memory_threshold,
                 num_actions,
                 num_features
                ):

        self.actor_critic = actor_critic
        self.actor_critic_target = actor_critic_target
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.soft_update_freq = soft_update_freq
        self.soft_update_step = soft_update_step
        self.batch_size = batch_size
        self.memory_threshold = memory_threshold
        self.num_actions = num_actions
        self.num_features = num_features

        self.TYPE_LAYER_NAME = "type_output"
        self.PARAM_LAYER_NAME = "param_output"
        self.WRITER_FILE_PATH = "tensorboard_report"

    def compile(self, optimizer, loss_func):
        """
          Compile online network (and target network if target fixing/ double Q-learning
            is being utilized) with provided optimizer/loss functions.
        """
        self.actor_critic.compile(loss=loss_func,optimizer=optimizer)
        self.actor_critic_target.compile(loss=loss_func,optimizer=optimizer)

    #TO_DO (NOT NECESSARY ANYMORE)
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

    def calc_action_vector(self, state):
        # state: numpy.array of the state(s)
        # action_vector: (batch size, 10). action type (4) comes first, followed by params (6) 
        get_type_output = K.function([self.actor_critic.layers[0].input],
                          [self.actor_critic.get_layer(self.TYPE_LAYER_NAME).output])
        type_output = get_type_output([state])[0]

        get_param_output = K.function([self.actor_critic.layers[0].input],
                          [self.actor_critic.get_layer(self.PARAM_LAYER_NAME).output])
        param_output = get_param_output([state])[0]

        action_vector = np.concatenate((type_output, param_output), axis=1)
        return action_vector


    
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

    #TO_DO: held out states
    def populate_replay_memory_and_held_out_states(self, env):
        """
        Populates replay memory and held out states memory with samples
        selected using a random policy.

        Parameters
        ----------
        env: HFOEnvironment
          Environment to be fitted. 
        """
        print("Populating replay mem ...")

        # populate replay memory
        cur_iteration = 0
        for episode in itertools.count():
            #indicate game has started
            status = IN_GAME
            #get initial state
            s_t = env.getState()
            s_t = np.asmatrix(s_t)

            first_time_kickablezone = 0
            while status == IN_GAME:
                if(cur_iteration == self.memory_threshold):
                    print("Done populating replay mem ...")
                    return

                #get action vector
                action_vector = self.calc_action_vector(s_t)

                #loads action ie: env.act(DASH, 20.0, 0.)
                self.load_action(env, action_vector)

                #advance the environment and get the game status
                [status, is_terminal] = self.advance_environment(env)

                #get new state
                s_t1 = env.getState()
                s_t1 = np.asmatrix(s_t1)

                #get reward
                [r_t, kickable_count] = self.get_reward(s_t, s_t1, action_vector,first_time_kickablezone,status)
                first_time_kickablezone = kickable_count

                #store sample
                self.memory.append(s_t, action_vector, r_t, s_t1, is_terminal)

                #set new current state
                s_t = s_t1

                cur_iteration+=1

        

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
    
    #TO_DO Batch normalization??
    def update_network(self):
        sample_batch = self.memory.sample(self.batch_size)

        #test forward/backward propegate for all networks
        #get inputs for actor/critc networks and rewards
        next_states = np.zeros((self.batch_size,self.num_features))
        current_states = np.zeros((self.batch_size,self.num_features))
        r_batch = np.zeros((self.batch_size,1))

        for i in range(len(sample_batch)):
            next_states[i] = sample_batch[i].s_t1
            current_states[i] = sample_batch[i].s_t
            r_batch[i] = sample_batch[i].r_t

        # forward pass to get Q(s',a')
        q_next = self.actor_critic_target.predict_on_batch(next_states)
        q_true = np.array(q_next)*self.gamma + r_batch
        loss = self.actor_critic.train_on_batch(current_states, q_true)
        return loss

    def soft_update_target(self):
        # do softupdate on the target network
        update_portion = self.soft_update_step
        online_weights = self.actor_critic.get_weights()
        online_weights = np.array(online_weights)
        old_target_weights = self.actor_critic_target.get_weights()
        old_target_weights = np.array(old_target_weights)

        new_target_weights = update_portion*online_weights + (1-update_portion)*old_target_weights
        self.actor_critic_target.set_weights(new_target_weights)


    def get_reward(self, s_t, s_t1, action_vector,first_time_kickablezone,status):
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
        reward = 0

        #get current ball dist
        ball_proxim_t1 = s_t1[0,53]
        ball_dist_t1 = 1.0-ball_proxim_t1

        #get previous ball dist
        ball_proxim_t = s_t[0,53]
        ball_dist_t = 1.0-ball_proxim_t

        #get change in distance
        ball_dist_delta = ball_dist_t - ball_dist_t1

        #Increment reward by change in ball distance from self
        reward += ball_dist_delta

        #get goal distance curr nd prev
        goal_proxim_t1 = s_t1[0,15]
        goal_dist_t1 = 1.0-goal_proxim_t1
        goal_proxim_t = s_t[0,15]
        goal_dist_t = 1.0-goal_proxim_t

        #get change in goal dist
        goal_dist_delta = goal_dist_t - goal_dist_t1

        # get if agent in current state is able to kick:
        kickable_t1 = s_t1[0,12]
        kickable_t = s_t[0,12]
        kickable_delta = kickable_t1 - kickable_t
        # give rewards for going into kickable zone the first time in an episode
        # Include additonal checks when more than one player
        
        if(first_time_kickablezone == 0 and kickable_delta >= 1):
            first_time_kickablezone += 1
            #Increment reward by 1
            reward += 1
        #calculate distance between ball and goal using cosine law
        # it's the 3rd side of the traingle formed with ball_dist and goal_dist 
        #### For curr state ######################
        ball_ang_sin_rad_t1 = s_t1[0,51]
        ball_ang_cos_rad_t1 = s_t1[0,52]
        #adjust range from (-pi,pi)
        ball_ang_rad_t1 = math.acos(ball_ang_cos_rad_t1)
        if (ball_ang_sin_rad_t1 < 0):
            ball_ang_rad_t1 *= -1.0
        goal_ang_sin_rad_t1 = s_t1[0,13]
        goal_ang_cos_rad_t1 = s_t1[0,14]
        goal_ang_rad_t1 = math.acos(goal_ang_cos_rad_t1)
        if (goal_ang_sin_rad_t1 < 0):
            goal_ang_rad_t1 *= -1.0

        alpha_t1 = max(ball_ang_rad_t1, goal_ang_rad_t1) - min(ball_ang_rad_t1, goal_ang_rad_t1)
        # By law of cosines. Alpha is angle between ball and goal
        ball_dist_goal_t1 = sqrt(ball_dist_t1*ball_dist_t1 + goal_dist_t1*goal_dist_t1 -
                              2.*ball_dist_t1*goal_dist_t1*cos(alpha_t1))
        
        ######## For previous state ######################################
        ball_ang_sin_rad_t = s_t[0,51]
        ball_ang_cos_rad_t = s_t[0,52]
        #adjust range from (-pi,pi)
        ball_ang_rad_t = math.acos(ball_ang_cos_rad_t)
        if (ball_ang_sin_rad_t < 0):
            ball_ang_rad_t *= -1.0
        goal_ang_sin_rad_t = s_t[0,13]
        goal_ang_cos_rad_t = s_t[0,14]
        goal_ang_rad_t = math.acos(goal_ang_cos_rad_t)
        if (goal_ang_sin_rad_t < 0):
            goal_ang_rad_t *= -1.0

        alpha_t = max(ball_ang_rad_t, goal_ang_rad_t) - min(ball_ang_rad_t, goal_ang_rad_t)
        # By law of cosines. Alpha is angle between ball and goal
        ball_dist_goal_t = sqrt(ball_dist_t*ball_dist_t + goal_dist_t*goal_dist_t -
                              2.*ball_dist_t*goal_dist_t*cos(alpha_t))

        #change in distance between ball and goal
        ball_dist_goal_delta = ball_dist_goal_t - ball_dist_goal_t1

        #incremenr reward by 3. change in distance between goal and ball
        reward += 3.0*ball_dist_goal_delta

        #Check if goal or not
        
        if(status == GOAL):
        #check for 'unexpected side'???
            reward += 5

        return reward, first_time_kickablezone

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
        
        #populate replay memory
        self.populate_replay_memory_and_held_out_states(env)

        #init metric vectors (to eventually plot)
        #TO_DO implement logging
        #loss_log=np.zeros(num_iterations)
        #reward_log=np.zeros(int(np.ceil(num_iterations/self.reward_samp)))

        #iterate through environment samples

        # writer for tensorboard
        # the path is a new subfolder created in the working dir
        writer = tf.summary.FileWriter(self.WRITER_FILE_PATH)


        cur_iteration = 0
        for episode in itertools.count():
            #indicate game has started
            status = IN_GAME
            #get initial state
            s_t = env.getState()
            s_t = np.asmatrix(s_t)
            first_time_kickablezone = 0
            while status == IN_GAME:
                if(cur_iteration == num_iterations):
                    #TO_DO save everything
                    #possibly save replay memory
                    return

                #update network
                loss = self.update_network()
                # updates the target network with information of the online network
                # only happens once in a while (according to soft_update_freq)
                if cur_iteration % self.soft_update_freq == 0:
                    self.soft_update_target()
            
                #get action vector
                action_vector = self.calc_action_vector(s_t)

                #loads action ie: env.act(DASH, 20.0, 0.)
                self.load_action(env, action_vector)

                #advance the environment and get the game status
                [status, is_terminal] = self.advance_environment(env)

                #get new state
                s_t1 = env.getState()
                s_t1 = np.asmatrix(s_t1)

                #get reward
                [r_t,kickable_count] = self.get_reward(s_t, s_t1, action_vector, first_time_kickablezone, status)
                first_time_kickablezone = kickable_count

                #store sample
                self.memory.append(s_t, action_vector, r_t, s_t1, is_terminal)

                #set new current state
                s_t = s_t1

                cur_iteration+=1

            # Check the outcome of the episode
            print('Episode %d ended with %s'%(episode, env.statusToString(status)))
            # Quit if the server goes down
            if status == SERVER_DOWN:
                env.act(QUIT)
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

    # call this function like this:
    # self.save_scalar(steps_after_first_memfull, 'avg_reward', avg_reward, writer)
    # step:iteration count (x axis of the plots)
    # name: name of variable to store in
    # value: value to be stored
    # writer: writer object
    def save_scalar(self, step, name, value, writer):
      """Save a scalar value to tensorboard.
      
      Parameters
      ----------
      step: int
        Training step (sets the position on x-axis of tensorboard graph.
      name: str
        Name of variable. Will be the name of the graph in tensorboard.
      value: float
        The value of the variable at this step.
      writer: tf.FileWriter
        The tensorboard FileWriter instance.
      """
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = float(value)
        summary_value.tag = name
        writer.add_summary(summary, step)
