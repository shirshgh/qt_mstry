import tensorflow as tf
from Board import TTTBoard


REPLAY_MEMORY = 20000
NUM_ITERATIONS = 100
hidden1 = 32
hidden2 = 32
"""
  Args:
    inputs: inputs placeholder.
    h1_w_units: Size of the first hidden layer.
    h2_w_units: Size of the second hidden layer.

  Returns:
    prob_out: 1D tensor with the actions probabilities.
  """

#input_flat - [batch_size, NxN]
def dqn_ttt(inputs, h1_w_units, h2_w_units):
	with tf.name_scope('hidden1'):
		weights = tf.Variable(tf.random_normal([inputs.get_shape().as_list()[-1], h1_w_units]),
			name='weights')
		biases = tf.Variable(tf.zeros([h1_w_units]),
			name='biases')
		hidden1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), biases))

	with tf.name_scope('hidden2'):
		weights = tf.Variable(tf.random_normal([h1_w_units, h2_w_units]),
            name='weights')
		biases = tf.Variable(tf.zeros([h2_w_units]),
            name='biases')
		hidden2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden1, weights), biases))

	with tf.name_scope('prob_out'):
		weights = tf.Variable(tf.random_normal([h2_w_units, inputs.get_shape().as_list()[-1]]),
			name='weights')
		biases = tf.Variable(tf.zeros([inputs.get_shape().as_list()[-1]]),
           name='biases')
		prob_out = tf.nn.softmax(tf.nn.bias_add(tf.matmul(hidden2, weights), biases))
	return prob_out

def loss(inputs, memory):
    #TBD by Maor
    return loss

def get_data():
  # Import data   # connect to API 
	return 3    
def main(n=3,k=3):
	
	board = TTTBoard(n,n,k)
	moves_p = ["1","1","1"] 
	moves_x = [0,1,2] 
	moves_y = [0,1,2] 
	for i in range(len(moves_p)):
		print(board.play(moves_p[i],moves_x[i],moves_y[i]))
		board.draw() 
	#memory = replay_memory(capcity=REPLAY_MEMORY)
 	# Create the model
	board_size = n*n
	x = tf.placeholder(tf.float32, [None, board_size])
    
    # Define loss

    # Build the graph for the deep net
	prob_out = dqn_ttt(x,hidden1,hidden2)
	y_ = tf.placeholder(tf.float32, [None, board_size])
	with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())


	return 



	for i in range(NUM_ITERATIONS):
		state, action, reward, next_state, done = board.play()
	
		with tf.name_scope('loss'):
			loss = loss(prob_out,memory)
		if not done:
			next_state = current_screen - last_screen
		else:
			next_state = None
		done = true

        # Store the transition in memory
		memory.push(state, action, next_state, reward, done)

        # Move to the next state
		state = next_state

if __name__ == "__main__":
	main()
