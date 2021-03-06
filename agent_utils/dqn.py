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
import h5py

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
                 num_features,
                 eval_freq,
                 episodes_per_eval,
                 save_freq
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
        self.eval_freq = eval_freq
        self.episodes_per_eval = episodes_per_eval
        self.save_freq = save_freq

        self.TYPE_LAYER_NAME = "type_output"
        self.PARAM_LAYER_NAME = "param_output"
        self.WRITER_FILE_PATH = "tensorboard_report"
        self.MODEL_FILE_STRING_AC = "ac_model_"
        self.MODEL_FILE_STRING_TAR = "ac_target_model_"
        self.MEM_FILE_NAME = "mem_replay_"

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
                [r_t, kickable_count,status] = self.get_reward(s_t, s_t1,first_time_kickablezone,status)
                first_time_kickablezone = kickable_count
                if(status != IN_GAME):
                    is_terminal = True

                #store sample
                self.memory.append(s_t, action_vector, r_t, s_t1, is_terminal)

                #set new current state
                s_t = s_t1

                cur_iteration+=1

        

    def advance_environment(self, env):
        status = env.step()

        is_terminal = False

        if status != IN_GAME:
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


    def get_reward(self,s_t, s_t1,first_time_kickablezone,status):
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
        #if status == GOAL:
          #print("prox curr: ", ball_proxim_t1)
        ball_dist_t1 = 1.0-ball_proxim_t1

        #get previous ball dist
        ball_proxim_t = s_t[0,53]
        #if status == GOAL:
          #print("prox prev: ", ball_proxim_t)
        ball_dist_t = 1.0-ball_proxim_t

        #get change in proximity
        ball_prox_delta = ball_proxim_t1 - ball_proxim_t
        ball_dist_delta = ball_dist_t - ball_dist_t1

        #print("ball dist curr: ", )

        #print("change in proximity ", ball_prox_delta)

        #Increment reward by change in ball distance from self

        #round off everything to 3 places
        ball_prox_delta
        reward += ball_prox_delta

        #if(status == GOAL):
          #print("reward at goal after adding 'move towards ball'  ", reward)

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
            reward += 1.0
            #print("got +1 reward for first time entering kickable region")
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
                              2.0*ball_dist_t*goal_dist_t*cos(alpha_t))
        #if(status == GOAL):
        #print("alpha curr ", alpha_t1*180/math.pi)
        #print("alpha prev", alpha_t*180/math.pi)
          

        #print("ball_distance to goal curr ", ball_dist_goal_t1)
        #print("ball_distance to goal prev ", ball_dist_goal_t)
        #change in distance between ball and goal

        ball_dist_goal_delta = ball_dist_goal_t - ball_dist_goal_t1
        #if(status == GOAL):
          #print("change in distance between ball and goal", ball_dist_goal_delta)

        


        #incremenr reward by 3. change in distance between goal and ball
        reward += 3.0*ball_dist_goal_delta
        #if(status == GOAL):
        #print("reward at goal after adding r for 'move ball towards goal' ", reward)
        
        #Check if goal or not
        if(ball_dist_goal_t < 0.1):
          status = GOAL
          reward = 5.0
          

        
        if(status == GOAL):
        #check for 'unexpected side'???
            #print("status", GOAL)
            #print ("reward before goal", reward)
            reward =  5.0
            #print ("reward after goal", reward)
        
        return reward, first_time_kickablezone, status

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
                    self.save_models(cur_iteration)
                    self.save_replay_memory(cur_iteration)
                    return

                #update network
                loss = self.update_network()

                # updates the target network with information of the online network
                # only happens once in a while (according to soft_update_freq)
                if cur_iteration % self.soft_update_freq == 0:
                    self.soft_update_target()

                #check if network needs to be evaluated
                if cur_iteration % self.eval_freq == 0:
                    ave_reward, ave_qvalue = self.evaluate(env, self.episodes_per_eval)
                    self.save_scalar(cur_iteration, 'reward', ave_reward, writer)
                    self.save_scalar(cur_iteration, 'q_value', ave_qvalue, writer)

                #chekc if models + replay memory need to be saved
                if cur_iteration % self.save_freq == 0 and cur_iteration != 0:
                    self.save_models(cur_iteration)
                    self.save_replay_memory(cur_iteration)

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
                [r_t, kickable_count, status] = self.get_reward(s_t, s_t1, first_time_kickablezone, status)
                first_time_kickablezone = kickable_count
                if(status != IN_GAME):
                    is_terminal = True


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
                print("SERVER DOWN...")
                self.save_models(cur_iteration)
                self.save_replay_memory(cur_iteration)
                break


    def evaluate(self, env, num_episodes):
        """
        Evaluate policy.
        """
        print('Start Evaluate')
        ave_rewards = np.zeros(num_episodes)
        ave_qvalues = np.zeros(num_episodes)

        for episode in range(num_episodes):
            #indicate game has started
            status = IN_GAME
            #get initial state
            s_t = env.getState()
            s_t = np.asmatrix(s_t)
            first_time_kickablezone = 0
            reward = 0
            q_value = 0
            steps = 0
            while status == IN_GAME:
                action_vector = self.calc_action_vector(s_t)
                q_value += self.actor_critic_target.predict(s_t)

                #loads action ie: env.act(DASH, 20.0, 0.)
                self.load_action(env, action_vector)

                #advance the environment and get the game status
                [status, is_terminal] = self.advance_environment(env)

                #get new state
                s_t1 = env.getState()
                s_t1 = np.asmatrix(s_t1)

                #get reward
                [r_t,kickable_count,status] = self.get_reward(s_t, s_t1, first_time_kickablezone, status)
                first_time_kickablezone = kickable_count

                reward+=r_t

                #set new current state
                s_t = s_t1

                #increment number of steps taken during episode    
                steps+=1

            ave_rewards[episode] = reward/float(steps)
            ave_qvalues[episode] = q_value/float(steps)
        print('End Evaluate')
        return np.mean(ave_rewards), np.mean(ave_qvalues)

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

    def save_models(self, step):
        new_model_file_string = self.MODEL_FILE_STRING_AC + str(step) + '.h5'
        new_target_model_file_string = self.MODEL_FILE_STRING_TAR + str(step) + '.h5'
        self.actor_critic.save(new_model_file_string)
        self.actor_critic.save(new_target_model_file_string)

    def save_replay_memory(self, step):
        memory_size = self.memory.max_size
        sample_width = self.num_features*2 + self.num_actions + 2
        curr_mem_array = np.zeros((1, sample_width))
        counter = 0

        memList = list(self.memory.M)
        for sampleIdx in range(len(memList)):
            sample=memList[sampleIdx]
            s_t = sample.s_t
            a_t = sample.a_t
            new_line = np.concatenate((s_t,a_t),axis=1)
            r_t = sample.r_t
            new_line = np.concatenate((new_line,np.asmatrix(r_t)),axis=1)
            s_t1 = sample.s_t1
            new_line = np.concatenate((new_line,s_t1),axis=1)
            if sample.is_terminal: 
                is_terminal = 1.0
            else:
                is_terminal = 0.0 
            new_line = np.concatenate((new_line,np.asmatrix(is_terminal)),axis=1)
            curr_mem_array = np.concatenate((curr_mem_array, new_line))
            

        string_name = self.MEM_FILE_NAME + str(step) + '.h5'
        h5f = h5py.File(string_name, 'w')
        h5_name = 'replay_mem_' + str(step)
        h5f.create_dataset(h5_name, data=curr_mem_array)


