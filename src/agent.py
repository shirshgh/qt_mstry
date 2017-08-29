import tensorflow as tf
import numpy as np
import os

class Agent:

	def __init__(self,name,n,k,hidden_1,hidden_2):

		self.n = n
		self.k = k
		self.name = name
		self.hidden_1 = hidden_1
		self.hidden_2 = hidden_2
		self.input = tf.placeholder(tf.float32, shape=(None,self.n*self.n))
		self.dqn = self.dqn_ttt(self.input,self.hidden_1,self.hidden_2)
		self.predict = tf.argmax(self.dqn,axis=1)

	def dqn_ttt(self,inputs, hidden_1, hidden_2):

		with tf.name_scope(str(self.name)):

			weights1 = tf.Variable(tf.random_normal([inputs.get_shape().as_list()[-1], self.hidden_1]),name='weights1')
			biases1  = tf.Variable(tf.zeros([self.hidden_1]),name='biases1')
			dqn = tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights1), biases1))

			weights2 = tf.Variable(tf.random_normal([self.hidden_1, hidden_2]),name='weights2')
			biases2  = tf.Variable(tf.zeros([hidden_2]),name='biases2')
			dqn = tf.nn.relu(tf.nn.bias_add(tf.matmul(dqn, weights2), biases2))

			weights3 = tf.Variable(tf.random_normal([self.hidden_2, inputs.get_shape().as_list()[-1]]), name='weights3')
			biases3  = tf.Variable(tf.zeros([inputs.get_shape().as_list()[-1]]), name='biases3')
			prob_out = tf.nn.softmax(tf.nn.bias_add(tf.matmul(dqn, weights3), biases3))

		return prob_out