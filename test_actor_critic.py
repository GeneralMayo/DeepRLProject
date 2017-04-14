
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import backend as K
import tensorflow as tf
import keras

init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
new_relu = keras.layers.advanced_activations.LeakyReLU(alpha=0.01)


def init_actor_model(init, new_relu):

    

    actor_input = Input(shape=(58,), name='actor_input') #beware of dtype
    actor_hidden1 = Dense(1024, kernel_initializer=init, activation=new_relu)(actor_input)
    actor_hidden2 = Dense(512, kernel_initializer=init, activation=new_relu)(actor_hidden1)
    actor_hidden3 = Dense(256, kernel_initializer=init, activation=new_relu)(actor_hidden2)
    actor_hidden4 = Dense(128, kernel_initializer=init, activation=new_relu)(actor_hidden3)

    type_output = Dense(4, activation=new_relu, name="type_output")(actor_hidden4)
    param_output = Dense(6, activation=new_relu, name="param_output")(actor_hidden4)

    actor_model = Model(inputs=[actor_input], outputs=[type_output, param_output])
    actor_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return actor_model


def init_critic_model(init, new_relu):
    critic_state_input = Input(shape=(58,), name='critic_state_input')#beware of dtype
    critic_action_input = Input(shape=(10,), name='critic_action_input')

    x = keras.layers.concatenate([critic_state_input, critic_action_input])

    critic_hidden1 = Dense(1024, kernel_initializer=init, activation=new_relu)(x)
    critic_hidden2 = Dense(512, kernel_initializer=init, activation=new_relu)(critic_hidden1)
    critic_hidden3 = Dense(256, kernel_initializer=init, activation=new_relu)(critic_hidden2)
    critic_hidden4 = Dense(128, kernel_initializer=init, activation=new_relu)(critic_hidden3)

    critic_output = Dense(1, activation=new_relu, name="critic_output")(critic_hidden4)

    critic_model = Model(inputs=[critic_state_input, critic_action_input], outputs=[critic_output])
    critic_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return critic_model




# actor_model = init_actor_model(init, new_relu)
# actor_model_target = init_actor_model(init, new_relu)

# critic_model = init_critic_model(init, new_relu)
# critic_model_target = init_critic_model(init, new_relu)


def init_actor_critic(init, new_relu):
    actor_input = Input(shape=(58,), name='actor_input') #beware of dtype
    actor_hidden1 = Dense(1024, kernel_initializer=init, activation=new_relu)(actor_input)
    actor_hidden2 = Dense(512, kernel_initializer=init, activation=new_relu)(actor_hidden1)
    actor_hidden3 = Dense(256, kernel_initializer=init, activation=new_relu)(actor_hidden2)
    actor_hidden4 = Dense(128, kernel_initializer=init, activation=new_relu)(actor_hidden3)

    type_output = Dense(4, activation=new_relu, name="type_output")(actor_hidden4)
    param_output = Dense(6, activation='sigmoid', name="param_output")(actor_hidden4)

    print(type(actor_input))
    print(type(type_output))
    print(type(param_output))
    x = keras.layers.concatenate([actor_input, type_output, param_output], axis=1)

    critic_hidden1 = Dense(1024, kernel_initializer=init, activation=new_relu)(x)
    critic_hidden2 = Dense(512, kernel_initializer=init, activation=new_relu)(critic_hidden1)
    critic_hidden3 = Dense(256, kernel_initializer=init, activation=new_relu)(critic_hidden2)
    critic_hidden4 = Dense(128, kernel_initializer=init, activation=new_relu)(critic_hidden3)

    critic_output = Dense(1, activation=new_relu, name="critic_output")(critic_hidden4)
    ac_model = Model(inputs=[actor_input], outputs=[critic_output])
    ac_model.compile(loss='mse', optimizer='adam')
    return ac_model

r = 1
gamma = 0.99
batch_size = 1
s = np.random.random((batch_size,58))
s_new = np.random.random((batch_size,58))
ac_model = init_actor_critic(init, new_relu)
print('old weights')
weights = ac_model.get_weights()
print(weights[0])

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
print('alalalla')





print('alalalla')
print('alalalla')






