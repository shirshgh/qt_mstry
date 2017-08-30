import tensorflow as tf
import argparse
import shutil
import numpy as np

from Board import TTTBoard,Status
from agent import Agent
from replay_memory import replay_memory

"""
  Args:
    inputs: inputs placeholder.
    h1_w_units: Size of the first hidden layer.
    h2_w_units: Size of the second hidden layer.

  Returns:
    prob_out: 1D tensor with the actions probabilities.
  """

def loss(inputs, memory):
    #TBD by Maor
    return loss

def main():
	
	parser = argparse.ArgumentParser()

	parser.add_argument('--n',				'-n',	type=int,	default=3,		help='Board size')
	parser.add_argument('--k',				'-k',	type=int,	default=3,		help='winning row')
	parser.add_argument('--hidden1',		'-h1',	type=int,	default=32,		help='hidden_layer 1')
	parser.add_argument('--hidden2',		'-h2',	type=int,	default=32,		help='hidden_layer 2')
	parser.add_argument('--replay_memory',	'-r',	type=int,	default=10000,	help='capacity')
	parser.add_argument('--num_iterations',	'-e',	type=int,	default=100,	help='num_iterations')
	parser.add_argument('--batch_size',		'-b',	type=int,	default=32,		help='batch size')
	parser.add_argument('--gamma',			'-g',	type=float,	default=0.9,	help='gamma')
	parser.add_argument('--lr',				'-l',	type=float,	default=1e-3,	help='gamma')
	args = parser.parse_args()

	p1 = Agent(1,args.n,args.k,args.hidden1,args.hidden2)
	#p2 = Agent(2,args.n,args.k,args.hidden1,args.hidden2)
	Skynet = 2
	board = TTTBoard(args.n,args.n,args.k)
	memory = replay_memory(args.replay_memory)
	done = False
	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())
		for i in range(args.replay_memory):

			if (done):
				board.reset()

			state = board.get()
			p1_Play,_ = sess.run(p1.dqn,feed_dict={p1.input:state})
			R = board.play(p1.name,p1_Play)
			done = (R != Status.SUCCESS) or board.game_over()

			if not done:
				R = board.play_random(Skynet)
				if (R == Status.WIN):
					R = Status.LOSE
				done = (R != Status.SUCCESS) or board.game_over()

			memory.push(state, p1_Play , R , board.get(), done)

		target_ph = tf.placeholder(tf.float32, shape=(None,))
		idx,Q_val = p1.dqn

		loss = tf.square(target_ph - Q_val)
		opt_step = tf.train.GradientDescentOptimizer(args.lr).minimize(loss)

		for _ in range(args.num_iterations):

			batch = memory.sample(args.batch_size)
			input = np.zeros([args.batch_size,args.n*args.n])
			target= np.zeros([args.batch_size])

			for i,sample in enumerate(batch):
				input[i,:] = sample.state
				if (sample.done):
					target[i] = sample.reward
				else:
					target[i] = sample.reward + args.gamma * (sess.run(Q_val,feed_dict={p1.input:sample.next_state}))

			sess.run(opt_step,feed_dict={target_ph:target,p1.input:input})

if __name__ == "__main__":
	main()
