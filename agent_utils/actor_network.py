
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers.normalization import BatchNormalization
import keras
from keras import backend as K

class ActorNetwork:
	def __init__(self, 
		sess, 
		num_states, 
		num_action_types, 
		num_action_params,
		RELU_NEG_SLOPE, 
		LEARNING_RATE,
		GRAD_CLIP):

		self.sess = sess

		init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

		#make target + online network ops
		self.online_state_input = tf.placeholder(tf.float32, shape=(None, num_states), name='online_state_input')
		self.online_network = self.init_actor(init, RELU_NEG_SLOPE, self.online_state_input)
		self.target_state_input = tf.placeholder(tf.float32, shape=(None, num_states), name='target_state_input')
		self.target_network = self.init_actor(init, RELU_NEG_SLOPE, self.target_state_input)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

		#train actor tensors/ops
		self.action_type_grad = tf.placeholder(tf.float32, shape=(None, num_action_types), name='action_type_grad')
		self.action_param_grad = tf.placeholder(tf.float32, shape=(None, num_action_params), name='action_param_grad')
		weight_grads = tf.gradients(self.online_network.outputs, self.online_network.weights, grad_ys=[self.action_type_grad, self.action_param_grad])
		def ClipIfNotNone(grad):
			if grad is None:
				return grad
			return tf.clip_by_value(grad, -GRAD_CLIP, GRAD_CLIP)

		capped_gvs = [ClipIfNotNone(grad) for grad in weight_grads]
		self.train_op = self.optimizer.apply_gradients(zip(capped_gvs, self.online_network.weights))

	def online_predict(self, state):
		return self.sess.run(self.online_network.output, {self.online_state_input: np.asmatrix(state),K.learning_phase(): 0})

	def target_predict(self, state):
		return self.sess.run(self.target_network.output, {self.target_state_input: np.asmatrix(state),K.learning_phase(): 0})

	def train(self, actor_scaled_output_grads, online_state_input_batch):
		self.sess.run(self.train_op, {
		    self.action_type_grad: actor_scaled_output_grads[0],
		    self.action_param_grad: actor_scaled_output_grads[1],
		    self.online_state_input: np.asmatrix(online_state_input_batch),
		    K.learning_phase(): 1
		    })

	def init_actor(self, init, RELU_NEG_SLOPE, actor_input_tensor):
	    actor_input = Input(tensor=actor_input_tensor)
	    actor_hidden1_Dense = Dense(1024, kernel_initializer=init)(actor_input)
	    actor_hidden1_Dense_Norm = BatchNormalization()(actor_hidden1_Dense)
	    actor_hidden1 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='1')(actor_hidden1_Dense_Norm)
	    actor_hidden2_Dense = Dense(512, kernel_initializer=init)(actor_hidden1)
	    actor_hidden2_Dense_Norm = BatchNormalization()(actor_hidden2_Dense)
	    actor_hidden2 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='2')(actor_hidden2_Dense_Norm)
	    actor_hidden3_Dense = Dense(256, kernel_initializer=init)(actor_hidden2)
	    actor_hidden3_Dense_Norm = BatchNormalization()(actor_hidden3_Dense)
	    actor_hidden3 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='3')(actor_hidden3_Dense_Norm)
	    actor_hidden4_Dense = Dense(128, kernel_initializer=init)(actor_hidden3)
	    actor_hidden4_Dense_Norm = BatchNormalization()(actor_hidden4_Dense)
	    actor_hidden4 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='4')(actor_hidden4_Dense_Norm)

	    type_output = Dense(4, kernel_initializer=init)(actor_hidden4)
	    param_output = Dense(6,kernel_initializer=init)(actor_hidden4)
	        
	    actor_model = Model(inputs=[actor_input], outputs=[type_output, param_output])

	    return actor_model