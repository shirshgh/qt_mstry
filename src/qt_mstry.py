import tensorflow as tf
import argparse
import shutil
import numpy as np
import os
import functools
import operator
import time
import sys

from board import TTTBoard,Status
from agent import Agent
from replay_memory import replay_memory
	
model_path =  '../data/my-model'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def leagal_move(move,n):
	try:
		move = int(move)
		return (move <=(n**2-1) and move >=0)
	except ValueError:
		return False

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
	print("Total parameters: " + str(total))

def load_and_play(board,p1,args):

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,model_path+"-"+str(args.load))
		
		done = False
		while not done:
			p1_Play = (sess.run(tf.argmax(p1.dqn,axis=1),feed_dict={p1.input:np.array(board.get())}))[0]
			R = board.play(p1.name,p1_Play)
			done = (R != Status.SUCCESS) or board.game_over()
			
			if not done:
				board.draw()

				move = sys.stdin.readline()
				while not leagal_move(move,args.n):
					print(str(move).rstrip() + " -- move is not in range [0," + str(args.n**2-1) + "]!")
					move = sys.stdin.readline()
				R = board.play(-1,int(move))
				done = (R != Status.SUCCESS) or board.game_over()

		print("Game Over")
		board.draw()

def observe(memory,sess,board,p,args):

	#p1,p2 = p_list
	iter = 0
	for i in range(args.observe_rounds):
		board.reset()
		for i in range(args.n**2):

			state_before = board.get()	
			p1_Play = p.play(sess,state_before)
			R = board.play(p.name,p1_Play)
			done = (R != Status.SUCCESS) or board.game_over()

			if not done:
				_,R = board.play_random(-1)
				if (R == Status.WIN):
					R = Status.LOSE
				done = (R != Status.SUCCESS) or board.game_over()

			state_after = board.get()	
			memory.push(state_before,p1_Play,R,state_after,done)
			if done:
				break

		if (iter % args.observe_cp == 0):
			print("Observe Round:\t" + str(iter) +  "\t" + str(len(memory)) + "\tepslion\t" + str(p.epslion))
		iter = iter + 1
	
	return memory

def replay(memory,sess,board,p1,args):

	saver = tf.train.Saver(max_to_keep=100)
	start = time.time()
	for i in range(args.replay_iter):

		p1.replay(sess,memory.sample(args.batch_size))
		if (i % args.replay_cp == 0):
			done = False
			board.reset()
			while not done:

				state = board.get()
				p1_Play = sess.run(p1.predict_play,feed_dict={p1.input:state})
				R = board.play(p1.name,p1_Play)
				done = (R != Status.SUCCESS) or board.game_over()
				if not done:
					_,R = board.play_random(-1)

					if (R == Status.WIN):
						R = Status.LOSE
					done = (R != Status.SUCCESS) or board.game_over()

			print("Replay  iteration\t" + str(p1.iter_count) +  "\tfinal reward:\t" + str(R) + "\ttime:\t" + "%.2f" % (time.time()-start))
			start = time.time()
			saver.save(sess, model_path,global_step = p1.iter_count)

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument('--n',				'-n',	type=int,				default=3,		help='Board size')
	parser.add_argument('--k',				'-k',	type=int,				default=3,		help='winning row')
	parser.add_argument('--hidden_1',		'-h1',	type=int,				default=64,		help='hidden_layer 1')
	parser.add_argument('--hidden_2',		'-h2',	type=int,				default=64,		help='hidden_layer 2')
	parser.add_argument('--hidden_3',		'-h3',	type=int,				default=64,		help='hidden_layer 2')
	parser.add_argument('--replay_memory',	'-r',	type=int,				default=50000,	help='capacity')
	parser.add_argument('--batch_size',		'-b',	type=int,				default=32,		help='batch size')
	parser.add_argument('--observe_cp',				type=int,				default=500,	help='')
	parser.add_argument('--observe_rounds',			type=int,				default=6000,	help='')
	parser.add_argument('--replay_cp',				type=int,				default=50,		help='')
	parser.add_argument('--replay_iter',			type=int,				default=500,	help='')
	parser.add_argument('--gamma',			'-g',	type=float,				default=0.95,	help='gamma')
	parser.add_argument('--lr',				'-l',	type=float,				default=1e-3,	help='learning rate')
	parser.add_argument('--load',					type=int,				default=0,		help='checkpoint to load from ')
	parser.add_argument('--print_variables',		action='store_true',	default=False,	help='checkpoint to load from ')
	args = parser.parse_args()

	p1 = Agent(1,args)
	if (args.print_variables):
		print_var_list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
	board = TTTBoard(args.n,args.n,args.k)
	
	#load trained model
	if args.load:
		return load_and_play(board,p1,args)

	#train new model
	memory = replay_memory(args.replay_memory)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(100):
			observe(memory,sess,board,p1,args)
			replay (memory,sess,board,p1,args)
	
if __name__ == "__main__":
	main()
