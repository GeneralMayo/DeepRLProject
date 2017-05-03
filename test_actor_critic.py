
import numpy as np
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input

import keras

init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

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


def init_critic(init, RELU_NEG_SLOPE, critic_input_tensor):
    critc_input = Input(tensor=critic_input_tensor)
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
    critic_model = Model(inputs=[critc_input], outputs=[critic_output])
    
    return critic_model


RELU_NEG_SLOPE = 0.01
LEARNING_RATE = .1
r = 1
gamma = 0.99
batch_size = 1
s = np.random.random((batch_size,58))
s_new = np.random.random((1,58))
SCALING_FACTOR = 10000000

#get actor model
a_state_input = tf.placeholder(tf.float32, shape=(None, 58), name='a_state_input')
actor_model = init_actor(init, RELU_NEG_SLOPE, a_state_input)

#get critic model
c_state_input = tf.placeholder(tf.float32, shape=(None, 58), name='c_state_input')
c_action_type_input = tf.placeholder(tf.float32, shape=(None, 4), name='c_action_type_input')
c_action_param_input = tf.placeholder(tf.float32, shape=(None, 6), name='c_action_param_input')
critic_input = tf.concat([c_state_input,c_action_type_input,c_action_param_input],axis=1)
critic_model = init_critic(init,RELU_NEG_SLOPE,critic_input)


adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

#train critic
loss1 = critic_model.output*2/batch_size
critic_train_op1 = adam.minimize(loss1)


#train actor
#get starting grads
actor_output_grads = tf.gradients(loss1, [c_action_type_input, c_action_param_input])
#scale parameter grads
actor_output_grads[1] = actor_output_grads[1]*SCALING_FACTOR

#update actor weights (Note: this is a new graph)
actor_grad_type_ph = tf.placeholder(tf.float32, shape=(None,4), name='actor_grad_type_ph')
actor_grad_param_ph = tf.placeholder(tf.float32, shape=(None,6), name='actor_grad_param_ph')
actor_weight_grads = tf.gradients(actor_model.outputs, actor_model.weights, grad_ys=[actor_grad_type_ph, actor_grad_param_ph])
actor_train_op = adam.apply_gradients(zip(actor_weight_grads, actor_model.weights))

init_op = tf.global_variables_initializer()
sess.run(init_op)

actor_output = sess.run(actor_model.output, {a_state_input: np.asmatrix(s)})
critic_output = sess.run(critic_model.output, 
        {c_state_input: np.asmatrix(s),
        c_action_type_input: np.asmatrix(actor_output[0]),
        c_action_param_input: np.asmatrix(actor_output[1])
        })
weightSum = 0
for wVec in critic_model.get_weights()[0]:
    print("HI")
    weightSum += np.sum(wVec)
print(weightSum)
input()
print(critic_output[0][0])
input()
sess.run(critic_train_op1, {
        c_state_input: np.asmatrix(s),
        c_action_type_input: np.asmatrix(actor_output[0]),
        c_action_param_input: np.asmatrix(actor_output[1])
        })

actor_grads = sess.run(actor_output_grads, {
        c_state_input: np.asmatrix(s),
        c_action_type_input: actor_output[0],
        c_action_param_input: actor_output[1]
        })


print(actor_model.layers[1].get_weights())

sess.run(actor_train_op, {
    actor_grad_type_ph: actor_grads[0],
    actor_grad_param_ph: actor_grads[1],
    a_state_input: np.asmatrix(s)
    })

print(actor_model.layers[1].get_weights())

"""
q_pred = ac_model.predict_on_batch(s)
q_pred = q_pred[0]
q_next = ac_model.predict_on_batch(s_new)
q_next = q_next[0]
print(type(q_next))
print(q_next)
print(s)
q_true = np.array(q_next)*gamma + r
ac_model.train_on_batch(s, q_true)

print('new weights')
weights = ac_model.get_weights()
print(weights[0])


new_pred = ac_model.predict(s,batch_size=batch_size)
"""




