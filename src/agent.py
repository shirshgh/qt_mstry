import tensorflow as tf
import numpy as np

class Agent:

    def __init__(self,n,k):
        
        self.n = n
        self.k = k

        self.input = tf.placeholder(tf.float32, shape=(self.n,self.n))
        self.W     = tf.get_variable("W",shape=(self.n**2,self.n**2))
        self.DQN   = tf.matmul(tf.reshape(self.input,[1,-1]),self.W)

    def Play(self, state):
        with tf.Session() as sess:  
            sess.run(tf.global_variables_initializer())       
            prob = sess.run(self.DQN,feed_dict={self.input:state})
        return np.argmax(prob)

