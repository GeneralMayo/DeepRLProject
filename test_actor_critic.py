
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import backend as K
import tensorflow as tf
import keras

init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)

def init_actor_critic_new(init,RELU_NEG_SLOPE):
    actor_input = Input(shape=(58,), name='actor_input') #beware of dtype
    actor_hidden1_Dense = Dense(1024, kernel_initializer=init)(actor_input)
    actor_hidden1 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='1')(actor_hidden1_Dense)
    actor_hidden2_Dense = Dense(512, kernel_initializer=init)(actor_hidden1)
    actor_hidden2 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='2')(actor_hidden2_Dense)
    actor_hidden3_Dense = Dense(256, kernel_initializer=init)(actor_hidden2)
    actor_hidden3 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='3')(actor_hidden3_Dense)

    actor_hidden4_Dense = Dense(128, kernel_initializer=init)(actor_hidden3)
    actor_hidden4 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE,name='4')(actor_hidden4_Dense)


    # Output of the actor network. this is still part of the actor-critic network in our architecture and 
    # these two layers will not be accessed as model output but intermediate layer output
    # Note that they both take inputs from the actor_hidden4 layer.
    type_output_Dense = Dense(4)(actor_hidden4)
    type_output = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE, name="type_output")(type_output_Dense)

    param_output_Dense = Dense(6, name="param_output")(actor_hidden4)
    param_output = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE)(param_output_Dense)

    x = keras.layers.concatenate([actor_input, type_output, param_output], axis=1)

    critic_hidden1_Dense = Dense(1024, kernel_initializer=init)(x)
    critic_hidden1 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE)(critic_hidden1_Dense)
    critic_hidden2_Dense = Dense(512, kernel_initializer=init)(critic_hidden1)
    critic_hidden2 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE)(critic_hidden2_Dense)
    critic_hidden3_Dense = Dense(256, kernel_initializer=init)(critic_hidden2)
    critic_hidden3 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE)(critic_hidden3_Dense)
    critic_hidden4_Dense = Dense(128, kernel_initializer=init)(critic_hidden3)
    critic_hidden4 = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE)(critic_hidden4_Dense)

    critic_output_Dense = Dense(1)(critic_hidden4)
    critic_output = keras.layers.advanced_activations.LeakyReLU(alpha=RELU_NEG_SLOPE, name="critic_output")(critic_output_Dense)
    ac_model = Model(inputs=[actor_input], outputs=[critic_output])
    return ac_model, param_output


RELU_NEG_SLOPE = 0.01
LEARNING_RATE = .0001
r = 1
gamma = 0.99
batch_size = 1
s = np.random.random((batch_size,58))
s_new = np.random.random((batch_size,58))

ac_model = init_actor_critic_new(init,RELU_NEG_SLOPE)
adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)




#set gradients of critic
#gradients = adam.compute_gradients(,)



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





