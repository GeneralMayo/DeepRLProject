
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import keras


class CriticNetwork:
  def __init__(self, 
    sess, 
    num_states, 
    num_action_types, 
    num_action_params,
    action_param_bounds,
    BATCH_SIZE, 
    RELU_NEG_SLOPE, 
    LEARNING_RATE):

    self.sess = sess

    init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

    self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    #online critic
    self.online_state_input = tf.placeholder(tf.float32, shape=(None, num_states), name='online_state_input')
    self.online_action_type_input = tf.placeholder(tf.float32, shape=(None, num_action_types), name='online_action_type_input')
    self.online_action_param_input = tf.placeholder(tf.float32, shape=(None, num_action_params), name='online_action_param_input')
    self.online_critic_input = tf.concat([self.online_state_input, self.online_action_type_input, self.online_action_param_input], axis=1)
    self.online_network = self.init_critic(init, RELU_NEG_SLOPE, self.online_critic_input)
    #target critic
    self.target_state_input = tf.placeholder(tf.float32, shape=(None, num_states), name='target_state_input')
    self.target_action_type_input = tf.placeholder(tf.float32, shape=(None, num_action_types), name='target_action_type_input')
    self.target_action_param_input = tf.placeholder(tf.float32, shape=(None, num_action_params), name='target_action_param_input')
    self.target_critic_input = tf.concat([self.target_state_input, self.target_action_type_input, self.target_action_param_input], axis=1)
    self.target_network = self.init_critic(init, RELU_NEG_SLOPE, self.target_critic_input)

    #loss ops (mean squared error)
    self.q_target = tf.placeholder("float",[None,1])
    self.loss = tf.pow(self.online_network.output-self.q_target,2)/BATCH_SIZE
    
    #train op
    self.train_op = self.optimizer.minimize(self.loss)

    #Scale Actor Grads
    #normal grads
    self.actor_output_grads = tf.gradients(self.online_network.output, [self.online_action_type_input, self.online_action_param_input])
    self.actor_output_grads[0] = self.actor_output_grads[0]/BATCH_SIZE
    self.actor_output_grads[1] = self.actor_output_grads[1]/BATCH_SIZE
    self.actor_parameter_output_grads = self.actor_output_grads[1]
    
    #scaling factors
    self.pmin = tf.constant(action_param_bounds[0], dtype = tf.float32)
    self.pmax = tf.constant(action_param_bounds[1], dtype = tf.float32)
    self.prange = tf.constant([x - y for x, y in zip(action_param_bounds[1], action_param_bounds[0])], dtype = tf.float32)
    self.scale_increasing = tf.div(-self.online_action_param_input + self.pmax, self.prange)
    self.scale_decreasing = tf.div( self.online_action_param_input - self.pmin, self.prange)

    #choose how to scale gradient 
    self.filter = tf.zeros([num_action_params])
    #if dQ/dP > 0 --> make sure parameter isn't INCREASED too much, else --> make sure parameter isn't DECREASED too much
    self.scaled_param_grads = tf.where(tf.greater(self.actor_parameter_output_grads, self.filter),
         tf.multiply(self.actor_parameter_output_grads, self.scale_increasing),
          tf.multiply(self.actor_parameter_output_grads, self.scale_decreasing))
    self.get_scaled_actor_output_grads = [self.actor_output_grads[0], self.scaled_param_grads]
  
  def train(self, q_target_batch, online_state_input_batch, online_action_type_input_batch, online_action_param_input_batch):
    self.sess.run(self.train_op, {self.q_target: q_target_batch,
        self.online_state_input: online_state_input_batch,
        self.online_action_type_input: online_action_type_input_batch,
        self.online_action_param_input: online_action_param_input_batch})

  def get_scaled_dQdP(self, online_state_input_batch,
        online_action_type_input_batch,
        online_action_param_input_batch):
    return self.sess.run(self.get_scaled_actor_output_grads, 
        {self.online_state_input: online_state_input_batch,
        self.online_action_type_input: online_action_type_input_batch,
        self.online_action_param_input: online_action_param_input_batch})


  def online_predict(self, state, action_type, action_param):
    return self.sess.run(self.online_network.output, 
        {self.online_state_input: state,
        self.online_action_type_input: action_type,
        self.online_action_param_input: action_param})

  def target_predict(self, state, action_type, action_param):
    return self.sess.run(self.target_network.output, {self.target_state_input: state, self.target_action_type_input: action_type,self.target_action_param_input: action_param})

  def init_critic(self, init, RELU_NEG_SLOPE, critic_input_tensor):
    critic_input = Input(tensor=critic_input_tensor)
    critic_hidden1_Dense = Dense(1024, kernel_initializer=init)(critic_input)
    critic_hidden1 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE)(critic_hidden1_Dense)
    critic_hidden2_Dense = Dense(512, kernel_initializer=init)(critic_hidden1)
    critic_hidden2 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE)(critic_hidden2_Dense)
    critic_hidden3_Dense = Dense(256, kernel_initializer=init)(critic_hidden2)
    critic_hidden3 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE)(critic_hidden3_Dense)
    critic_hidden4_Dense = Dense(128, kernel_initializer=init)(critic_hidden3)
    critic_hidden4 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE)(critic_hidden4_Dense)

    critic_output_Dense = Dense(1)(critic_hidden4)
    critic_output = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE, name="critic_output")(critic_output_Dense)
    critic_model = Model(inputs=[critic_input], outputs=[critic_output])
    
    return critic_model
