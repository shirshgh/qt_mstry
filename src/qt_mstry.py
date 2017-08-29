import tensorflow as tf
from replay_memory import tranistion
REPLAY_MEMORY = 20000
NUM_ITERATIONS = 100

"""
  Args:
    inputs: inputs placeholder.
    h1_w_units: Size of the first hidden layer.
    h2_w_units: Size of the second hidden layer.

  Returns:
    prob_out: 1D tensor with the actions probabilities.
  """

#input_flat - [batch_size, NxN]
def dqn_ttt(inputs, h1_w_units, h2_w_units)
	with tf.name_scope('hidden1'):
		weights = tf.Varible(tf.random_normal([inputs.get_shape().to_list()[-1], h1_w_units]),
			name='weights')
		biases = tf.Variable(tf.zeros([h1_w_units]),
			name='biases')
		hidden1 = tf.nn.relu(tf.bias_add(tf.matmul(inputs, weights), biases))

    with tf.name_scope('hidden2'):
		weights = tf.Varible(tf.random_normal([h1_w_units, h2_w_units]),
            name='weights')
		biases = tf.Variable(tf.zeros([h2_w_units]),
            name='biases')
		hidden2 = tf.nn.relu(tf.bias_add(tf.matmul(hidden1, weights), biases))

    with tf.name_scope('prob_out'):
        weights = tf.Varible(tf.random_normal([h2_w_units, inputs.get_shape().to_list()[-1]),
			name='weights')
        biases = tf.Variable(tf.zeros([inputs.get_shape().to_list()[-1]),
           name='biases')
		prob_out = tf.softmax(tf.bias_add(tf.matmul(hidden2, weight), biases))
    return prob_out

def loss(inputs, memory)
    #TBD by Maor
    return loss

def get_data()
  # Import data   # connect to API 

def main():
	memory = replay_memory(capcity=REPLAY_MEMORY)
 	# Create the model
    x = tf.placeholder(tf.float32, [None, 32])

    # Define loss

    # Build the graph for the deep net
	prob_out = dqn_ttt(x)

	with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
 
	for i in range(NUM_ITERATIONS)
		'state', 'action', 'reward', 'next_state', 'done' = board.play()
	
		with tf.name_scope('loss'):
			loss = loss(inputs=prob_out,
				memory)
		if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
	    done = true

        # Store the transition in memory
        memory.push(state, action, next_state, reward, done)

        # Move to the next state
        state = next_state