import tensorflow as tf
import argparse
import numpy as np
import os
import functools
import operator
import time
import matplotlib.pyplot as plt

from board import TTTBoard,Status
from agent import DQNAgent,RandomAgent,HumanAgent

model_path =  '../data/my-model'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
X_player = 1
O_Player = -1

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

def test(players,sess,board,args):

	tie = win_x = win_o = 0
	for i in range(args.test_rounds):
		board.reset()
		
		for i in range(args.n**2):
			p = players[i%2]
			move = p.play(sess,board.get())
			R = board.play(p.name,int(move))
			done = (R != Status.SUCCESS) or board.game_over()
			if done:
				if (R == Status.SUCCESS):
					tie = tie + 10
				if (R == Status.WIN):
					if (i%2 == 0):
						win_x = win_x + 1
					else:
						win_o = win_o + 1
				if (R == Status.INVALID_MOVE):
					if (i%2 == 0):
						win_o = win_o + 1
					else:
						win_x = win_x + 1
				break
	
	return [(x/args.test_rounds) for x in [win_x,tie,win_o]]

def load_and_play(board,players,args):

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,model_path)
		board.reset()
		for i in range(args.n**2):
			p = players[i%2]
			move = p.play(sess,board.get())
			R = board.play(p.name,int(move))
			done = (R != Status.SUCCESS) or board.game_over()
			board.draw()
			if done:
				break
		print("Game Over")

def observe(p1,p2,sess,board,args):

	p_list = [p1,p2]
	for j in range(args.observe_rounds):
		board.reset()
		for i in range(args.n**2):

			player = p_list[i%2]
			if i >0:
				prev_player = p_list[(i-1)%2]

			state_before = board.get()	
			move = player.train_play(sess,state_before)
			R = board.play(player.name,move)
			done = (R != Status.SUCCESS) or board.game_over()

			if i > 0:
				if (R == Status.SUCCESS):
					prev_reward = Status.SUCCESS
				elif (R == Status.WIN):
					prev_reward = Status.LOSE
				elif (R == Status.INVALID_MOVE):
					prev_reward = Status.SUCCESS
				prev_player.memorize(prev_tuple[0],prev_tuple[1],prev_reward,board.get(),done)

			prev_tuple = (state_before,move,R)
			if done:
				player.memorize(state_before,move,R,None,True)
				break

	return max(player.memory_length(),prev_player.memory_length())

def replay(p1,p2,sess,board,args):

	saver = tf.train.Saver(max_to_keep=20)
	start = time.time()
	p_list = [p1,p2]
	for i in range(args.replay_iter):
		p1.replay(sess,p1.sample_memory(args.batch_size))
		p2.replay(sess,p2.sample_memory(args.batch_size))
	saver.save(sess, model_path)

def plot_graph(stats):

	wins = []
	ties = []
	losses = []
	for stat in stats:
		wins.append(stat[0])
		ties.append(stat[1])
		losses.append(stat[2])

	plt.plot(wins,label="Wins")
	plt.plot(ties,label="Ties")
	plt.plot(losses,label="Losses")
	plt.legend()
	plt.ylabel("Win %")
	plt.xlabel("round")
	plt.show()	

def main(args):

	X_DQN    =    DQNAgent(X_player ,args)
	O_DQN    =    DQNAgent(O_Player ,args)
	X_Human  =  HumanAgent(X_player ,args)
	O_Human  =  HumanAgent(O_Player ,args)
	X_Random = RandomAgent(X_player ,args)
	O_Random = RandomAgent(O_Player ,args)

	if (args.print_vars):
		print_var_list(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

	board = TTTBoard(args.n,args.n,args.k)
	p1 = X_DQN 
	p2 = O_Random

	if (args.load):
		print("Play!")
		load_and_play(board,[X_DQN,O_Human],args)
		return

	stats = []
	with tf.Session() as sess:

		print("Training for " + str(args.episodes) + " episodes")
		sess.run(tf.global_variables_initializer())

		for i in range(args.episodes):

			start = time.time()
			mem_length = observe(p1,p2,sess,board,args)
			replay(p1,p2,sess,board,args)
			stats.append(test([p1,p2],sess,board,args))
			end = time.time()
			print("Episode " + str(i+1) + 
				  "\treplay memory length " +  str(mem_length) + 
				  "\treplay iter: " + str(p1.iter_count) + 
				  "\tX Winning rate " + "%.2f" % (stats[-1][0]) + 
				  "\tTime: " + "%.2f" % (end-start))

	if (args.plot_graph):
		plot_graph(stats)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--n',				'-n',	type=int,				default=3,		help='Board size')
	parser.add_argument('--k',				'-k',	type=int,				default=3,		help='winning row')
	parser.add_argument('--hidden_1',		'-h1',	type=int,				default=64,		help='hidden_layer 1')
	parser.add_argument('--replay_memory',	'-r',	type=int,				default=10000,	help='memory capacity')
	parser.add_argument('--batch_size',		'-b',	type=int,				default=64,		help='batch size')
	parser.add_argument('--observe_rounds',			type=int,				default=500,	help='observation rounds')
	parser.add_argument('--replay_iter',			type=int,				default=100,	help='replay from memory iterations')
	parser.add_argument('--test_rounds',			type=int,				default=1000,	help='number of rounds to evaluate player')
	parser.add_argument('--gamma',			'-g',	type=float,				default=0.99,	help='gamma')
	parser.add_argument('--epsilon_decay',			type=float,				default=0.99995,help='epsilon_decay')
	parser.add_argument('--min_epsilon',			type=float,				default=0.2,	help='min_epsilon')
	parser.add_argument('--lr',				'-l',	type=float,				default=0.001,	help='learning rate')
	parser.add_argument('--episodes',				type=int,				default=20,		help='observe/replay episodes')
	parser.add_argument('--print_vars',				action='store_true',	default=False,	help='print the TensorFlow variables')
	parser.add_argument('--plot_graph',				action='store_true',	default=False,	help='print the TensorFlow variables')
	parser.add_argument('--load',					action='store_true',	default=False,	help='print the TensorFlow variables')
	args = parser.parse_args()
	main(args)