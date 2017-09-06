import tensorflow as tf
import numpy as np
import random

class Agent:

	def __init__(self,name,args):

		self.n = args.n
		self.k = args.k
		self.name = name
		self.hidden_1 = args.hidden_1
		self.hidden_2 = args.hidden_2
		self.hidden_3 = args.hidden_3
		self.gamma = args.gamma
		self.batch_size = args.batch_size
		self.lr = args.lr

		self.input     		= tf.placeholder(tf.float32, shape=(1,self.n*self.n))
		self.action    		= tf.placeholder(tf.int32  , shape=())
		self.target    		= tf.placeholder(tf.float32, shape=())
		self.dqn			= self.dqn_ttt()
		self.opt 			= tf.train.AdamOptimizer(self.lr)
		self.predict 		= tf.gather(self.dqn,self.action,axis=1)
		self.loss			= tf.square(self.target - self.predict)
		self.opt_step  		= self.opt.minimize(self.loss)
		self.predict_play	= tf.argmax(self.dqn,axis=1)
		self.max_Q			= tf.reduce_max(self.dqn)
		self.count = 0

	def play(self,sess,state):

		if np.random.rand() <= 0.5:
			move = random.choice((np.where(state[0] == 0)[0]))
		else:
			move = random.randint(0,self.n**2-1)
		return move

	def replay(self,sess,batch):

		for i,sample in enumerate(batch):
			state, action, reward, next_state, done  = sample
			if done:
				target = reward
			else:
				target = reward + self.gamma * sess.run(self.max_Q,feed_dict = {self.input:next_state})

			sess.run(self.opt_step,feed_dict={	self.input:state,
												self.action:action,
												self.target:target})

	def dqn_ttt(self):

		with tf.name_scope(str(self.name)):
			
			w_std = 0.1
			b_std = 0.01
			depth = [self.input.get_shape().as_list()[-1], self.hidden_1, self.hidden_2, self.hidden_3, self.input.get_shape().as_list()[-1]]

			weights1 = tf.Variable(tf.random_normal([depth[0],depth[1]],stddev=w_std), name='weights1')
			weights2 = tf.Variable(tf.random_normal([depth[1],depth[2]],stddev=w_std), name='weights2')
			weights3 = tf.Variable(tf.random_normal([depth[2],depth[3]],stddev=w_std), name='weights3')
			weights4 = tf.Variable(tf.random_normal([depth[3],depth[4]],stddev=w_std), name='weights4')

			bias1 = tf.Variable(tf.random_normal([depth[1]],stddev=b_std), name='bias1')
			bias2 = tf.Variable(tf.random_normal([depth[2]],stddev=b_std), name='bias2')
			bias3 = tf.Variable(tf.random_normal([depth[3]],stddev=b_std), name='bias3')
			bias4 = tf.Variable(tf.random_normal([depth[4]],stddev=b_std), name='bias4')

			dqn = self.input 

			dqn = tf.matmul(dqn, weights1)
			dqn = tf.nn.bias_add(dqn, bias1)
			dqn = tf.nn.relu(dqn)

			dqn = tf.matmul(dqn, weights2)
			dqn = tf.nn.bias_add(dqn, bias2)
			dqn = tf.nn.relu(dqn)

			dqn = tf.matmul(dqn, weights3)
			dqn = tf.nn.bias_add(dqn, bias3)
			dqn = tf.nn.relu(dqn)

			dqn = tf.matmul(dqn, weights4)
			dqn = tf.nn.bias_add(dqn, bias4)

		return dqn