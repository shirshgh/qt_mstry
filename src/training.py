import tensorflow as tf
import numpy as np

GAMMA = 1e - 4
BATCH_SIZE = 64

# Define loss and optimizer
loss = tf.reduce_mean(reward + GAMMA * np.amax(network(next_state)[0]) - network(state))
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

#training
minibatch = random.sample(memory, batch_size)
for state, action, reward, next_state, done in minibatch:
    sess.run(optimizer, feed_dict={reward: reward, state: state, next_state: next_state})



''''''''''''''''''''''''''''''''''''''''''''''''''''''its only a draft - don't worry :) :) '''''''''''''''''''''''''''''''''''''







