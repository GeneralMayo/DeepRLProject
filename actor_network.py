
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input

class ActorNetwork:
	def __init__(self, 
		sess, 
		num_states, 
		num_action_types, 
		num_action_params, 
		RELU_NEG_SLOPE, 
		LEARNING_RATE):

		self.sess = sess

		self.state_input = tf.placeholder(tf.float32, shape=(None, num_states), name='state_input')
		self.model = init_actor(init, RELU_NEG_SLOPE, self.state_input)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

		#train actor tensors/ops
		self.action_type_grad = tf.placeholder(tf.float32, shape=(None, num_action_types), name='action_type_grad')
		self.action_param_grad = tf.placeholder(tf.float32, shape=(None, num_action_params), name='action_param_grad')
		weight_grads = tf.gradients(self.model.outputs, self.model.weights, grad_ys=[self.action_type_grad, self.action_param_grad])
		self.train_op = self.optimizer.apply_gradients(zip(weight_grads, self.model.weights))


	def train(self, ):
		sess.run(self.train_op, {
	    actor_grad_type_ph: actor_grads[0],
	    actor_grad_param_ph: actor_grads[1],
	    a_state_input: np.asmatrix(s)
	    })

	def init_actor(init, RELU_NEG_SLOPE, actor_input_tensor):
	    actor_input = Input(tensor=actor_input_tensor)
	    actor_hidden1_Dense = Dense(1024, kernel_initializer=init)(actor_input)
	    actor_hidden1 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='1')(actor_hidden1_Dense)
	    actor_hidden2_Dense = Dense(512, kernel_initializer=init)(actor_hidden1)
	    actor_hidden2 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='2')(actor_hidden2_Dense)
	    actor_hidden3_Dense = Dense(256, kernel_initializer=init)(actor_hidden2)
	    actor_hidden3 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='3')(actor_hidden3_Dense)
	    actor_hidden4_Dense = Dense(128, kernel_initializer=init)(actor_hidden3)
	    actor_hidden4 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='4')(actor_hidden4_Dense)

	    type_output_Dense = Dense(4)(actor_hidden4)
	    type_output = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE, name="type_output")(type_output_Dense)
	    param_output_Dense = Dense(6)(actor_hidden4)
	    param_output = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE, name="param_output")(param_output_Dense)
	        
	    actor_model = Model(inputs=[actor_input], outputs=[type_output, param_output])

	    return actor_model