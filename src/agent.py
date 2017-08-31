import tensorflow as tf
import numpy as np
import os
import random
import time

class Agent:

	def __init__(self,name,n,k,hidden_1,hidden_2,epsilon,epsilon_decay,epsilon_min,gamma,lr,batch_size):

		self.n = n
		self.k = k
		self.name = name
		self.hidden_1 = hidden_1
		self.hidden_2 = hidden_2
		self.hidden_3 = hidden_2+1
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.gamma = gamma
		self.batch_size = batch_size
		self.lr = lr

		self.input      = tf.placeholder(tf.float32, shape=(1,self.n*self.n))
		self.dqn		= self.dqn_ttt()
		self.opt 		= tf.train.GradientDescentOptimizer(self.lr)
		
	def play(self,sess,state):
	
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.n**2)
		else:
			move = sess.run(tf.argmax(self.dqn,axis=1),feed_dict = {self.input:state})[0]
			return move
		
	def replay(self,sess,batch):
		
		for i,sample in enumerate(batch):
		
			state, action, reward, next_state, done  = sample
			if done:
				target = reward
			else:
				target = reward + self.gamma * sess.run(tf.reduce_max(self.dqn),feed_dict = {self.input:next_state})
			cur_prediction	= tf.gather(self.dqn,action,axis=1)
			loss 			= tf.square(target - cur_prediction)
			sess.run(self.opt.minimize(loss),feed_dict={self.input:state})

		if self.epsilon > self.epsilon_min:
			self.epsilon = self.epsilon * self.epsilon_decay

	def dqn_ttt(self):

		with tf.name_scope(str(self.name)):
			std = 0.001
			weights1 = tf.Variable(tf.random_normal([self.input.get_shape().as_list()[-1], self.hidden_1],stddev=std),name='weights1')
			biases1  = tf.Variable(tf.zeros([self.hidden_1]),name='biases1')
			dqn = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.input, weights1), biases1))

			weights2 = tf.Variable(tf.random_normal([self.hidden_1, self.hidden_2],stddev=std),name='weights2')
			biases2  = tf.Variable(tf.zeros([self.hidden_2]),name='biases2')
			dqn = tf.nn.relu(tf.nn.bias_add(tf.matmul(dqn, weights2), biases2))

			weights3 = tf.Variable(tf.random_normal([self.hidden_2, self.hidden_3],stddev=std),name='weights3')
			biases3  = tf.Variable(tf.zeros([self.hidden_3]),name='biases3')
			dqn = tf.nn.relu(tf.nn.bias_add(tf.matmul(dqn, weights3), biases3))
			
			weights4 = tf.Variable(tf.random_normal([self.hidden_3, self.input.get_shape().as_list()[-1]],stddev=std), name='weights4')
			biases4  = tf.Variable(tf.zeros([self.input.get_shape().as_list()[-1]]), name='biases4')
			dqn = tf.nn.bias_add(tf.matmul(dqn, weights4), biases4)

		return dqn