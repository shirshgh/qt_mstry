import tensorflow as tf
import argparse
import shutil

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
	parser.add_argument('--replay_memory',	'-r',	type=int,	default=10,	help='capacity')
	parser.add_argument('--num_iterations',	'-e',	type=int,	default=100,	help='num_iterations')
	args = parser.parse_args()

	p1 = Agent(1,args.n,args.k,args.hidden1,args.hidden2)
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
			p1_Play = sess.run(p1.predict,feed_dict={p1.input:board.get()})
			R = board.play(p1.name,p1_Play)
			done = (R != Status.SUCCESS)

			if not done:
				R = board.play_random(Skynet)
				done = (R != Status.SUCCESS)

			memory.push(state, p1_Play , R , board.get(), done)

	print(len(memory))
	memory.printall()
if __name__ == "__main__":
	main()
