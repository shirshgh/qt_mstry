import tensorflow as tf
import numpy as np
import random
import sys
from replay_memory import replay_memory

def human_leagal_move(move,n):
	try:
		move = int(move)
		return (move <=(n**2-1) and move >=0)
	except ValueError:
		return False

class Agent:

	def __init__(self,name,args):
		self.n = args.n
		self.k = args.k 
		self.name = name
		self.iter_count = 0

	def replay(self,args,batch,size):
		return
		
class DQNAgent(Agent):

	def __init__(self,name,args):
		Agent.__init__(self,name,args)
		
		self.n = args.n
		self.nn = args.n**2
		self.k = args.k
		self.name = name
		self.memory = replay_memory(args.replay_memory)
		self.hidden_1 = args.hidden_1 
		self.gamma = args.gamma
		self.batch_size = args.batch_size
		self.lr = args.lr

		self.input     		= tf.placeholder(tf.float32, shape=(1,self.nn))
		self.action    		= tf.placeholder(tf.int32  , shape=())
		self.target    		= tf.placeholder(tf.float32, shape=())
		self.dqn			= self.dqn_ttt()
		self.opt 			= tf.train.AdamOptimizer(self.lr)
		self.predict 		= tf.gather(self.dqn,self.action,axis=1)
		self.loss			= tf.square(self.target - self.predict)
		self.opt_step  		= self.opt.minimize(self.loss)
		self.predict_play	= tf.argmax(self.dqn,axis=1)
		self.max_Q			= tf.reduce_max(self.dqn)
		
		self.epsilon = 1.0
		self.epsilon_decay = args.epsilon_decay
		self.min_epsilon = args.min_epsilon

	def train_play(self,sess,state):

		if np.random.rand() <= self.epsilon:
			if np.random.rand() <= 0.5:
				move = random.choice((np.where(state[0] == 0)[0]))
			else:
				move = random.randint(0,self.nn-1)
		else:
			move = sess.run(self.predict_play,feed_dict = {self.input:state})[0]

		if (self.epsilon > self.min_epsilon):
			self.epsilon = self.epsilon * self.epsilon_decay
		return move
	
	def memorize(self,*tuple):
		self.memory.push(*tuple)
	
	def memory_length(self):
		return len(self.memory)

	def play(self,sess,state):
		return sess.run(self.predict_play,feed_dict = {self.input:state})[0]

	def replay(self,sess,batch):
		self.iter_count = self.iter_count + 1
		for i,sample in enumerate(batch):
			state, action, reward, next_state, done  = sample
			if done:
				target = reward
			else:
				target = reward + self.gamma * sess.run(self.max_Q,feed_dict = {self.input:next_state})

			sess.run(self.opt_step,feed_dict={self.input:state,self.action:action,self.target:target})

	def sample_memory(self,batch):
		return self.memory.sample(batch)

	def dqn_ttt(self):

		init = tf.truncated_normal_initializer(stddev=0.5)
		with tf.variable_scope("Player_"+str(self.name)):
			dqn = tf.layers.dense(self.input,self.hidden_1,activation=tf.nn.relu, kernel_initializer = init, name="FC1")
			dqn = tf.layers.dense(dqn       ,self.nn	  ,activation=None      , kernel_initializer = init, name="out")
		return dqn

class RandomAgent(Agent):

	def __init__(self,name,args):
		Agent.__init__(self,name,args)
		self.epsilon = 0

	def play(self,sess,state):
		return random.choice((np.where(state[0] == 0)[0]))
	
	def train_play(self,sess,state):
		return self.play(sess,state)

	def replay(self,args,batch):
		self.iter_count = self.iter_count + 1
		return 

	def memorize(self,*tuple):
		return

	def memory_length(self):
		return 0
	
	def sample_memory(self,batch):
		return None
	
	def predict_move(self,sess,state):
		return self.play(None,state)

class HumanAgent(Agent):

	def __init__(self,name,args):
		Agent.__init__(self,name,args)

	def play(self,sess,state):
		move = sys.stdin.readline()
		while not human_leagal_move(move,self.n):
			print(str(move).rstrip() + " -- move is not in range [0," + str(self.n**2-1) + "]!")
			move = sys.stdin.readline()
		return move