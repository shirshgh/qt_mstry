import tensorflow as tf
import argparse
import shutil
import numpy as np
import os
import functools
import operator
import time

from Board import TTTBoard,Status
from agent import Agent
from replay_memory import replay_memory

def print_var_list(var_list):
	total = 0
	for i in var_list:
		if (len(i.get_shape().as_list())) == 0:
			num_param_in_var = 1
		else:
			num_param_in_var = functools.reduce(operator.mul,i.get_shape().as_list())
		strr = i.name + "\tParams: " + str(num_param_in_var)
		print(strr)
		total = total + num_param_in_var
	print("Total: " + str(total))

def encode_state(state):
	sum = 0
	state = state[0]
	for idx,bit in enumerate(state):
		sum = sum + (2**idx)*bit
	return sum
def main():

	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'	

	parser = argparse.ArgumentParser()

	parser.add_argument('--n',				'-n',	type=int,	default=3,		help='Board size')
	parser.add_argument('--k',				'-k',	type=int,	default=3,		help='winning row')
	parser.add_argument('--hidden1',		'-h1',	type=int,	default=32,		help='hidden_layer 1')
	parser.add_argument('--hidden2',		'-h2',	type=int,	default=32,		help='hidden_layer 2')
	parser.add_argument('--epsilon',				type=int,	default=1.0,	help='epsilon')
	parser.add_argument('--epsilon_decay',			type=int,	default=0.99,	help='epsilon decay')
	parser.add_argument('--epsilon_min',			type=int,	default=1e-2,	help='epsilon min')
	parser.add_argument('--replay_memory',	'-r',	type=int,	default=1000,	help='capacity')
	parser.add_argument('--num_iterations',	'-e',	type=int,	default=500,	help='num_iterations')
	parser.add_argument('--batch_size',		'-b',	type=int,	default=16,		help='batch size')
	parser.add_argument('--checkpoint',				type=int,	default=50,		help='batch size')
	parser.add_argument('--gamma',			'-g',	type=float,	default=0,	help='gamma')
	parser.add_argument('--lr',				'-l',	type=float,	default=1e-2,	help='learning rate')
	args = parser.parse_args()

	p1 = Agent(1,args.n,args.k,args.hidden1,args.hidden2,args.epsilon,args.epsilon_decay,args.epsilon_min,args.gamma,args.lr,args.batch_size)
	print_var_list(tf.get_collection(tf.GraphKeys.VARIABLES))

	Skynet = -1
	board = TTTBoard(args.n,args.n,args.k)
	memory = replay_memory(args.replay_memory)
	done = False
	Rs = []
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())
		
		for i in range(args.num_iterations):
		
			board.reset()
			sum = 0
			for _ in range(args.n**2):
				
				state_before = board.get()	
				
				p1_Play = p1.play(sess,state_before)
				R = board.play(p1.name,p1_Play)
				sum = sum + R
				done = (R != Status.SUCCESS) or board.game_over()
				
				if not done:
					_,R = board.play_random(Skynet)
					if (R == Status.WIN):
						R = Status.LOSE
						sum = sum + R
					done = (R != Status.SUCCESS) or board.game_over()
				
				state_after = board.get()	
				memory.push(state_before,p1_Play,R,state_after,done)

				print(sum)
				if done:
					break
			

			Rs.append(sum)
			#print(str(sum)+"\t"+ str(p1.epsilon)+"\t"+str(len(memory)))
			if len(memory) > args.batch_size:	
				p1.replay(sess,memory.sample(args.batch_size))

		import matplotlib.pyplot as plt
		plt.plot((Rs),label="train")
		plt.ylabel('log loss')
		plt.xlabel('iterations')
		plt.legend()
		plt.show()
		
		return
		##test!
		for i in range(100):
			
			done = False
			board.reset()
			while not done:
		
				state = board.get()
				p1_Play,_ = sess.run(p1.dqn,feed_dict={p1.input:state})
				print("Player played " + str(p1_Play))
				R = board.play(p1.name,p1_Play)
				done = (R != Status.SUCCESS) or board.game_over()
	
				if not done:
					_,R = board.play_random(Skynet)
					if (R == Status.WIN):
						R = Status.LOSE
					done = (R != Status.SUCCESS) or board.game_over()
			
			if (R == Status.WIN):
				board.draw()
			print("Game over - " + str(R))
			time.sleep(3)
			
if __name__ == "__main__":
	main()
