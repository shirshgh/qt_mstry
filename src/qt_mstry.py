import tensorflow as tf


#model for the ld the MNIST model up to where it may be used for inference.

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
        weights = tf.Varible(h1_w_units,
                              name='weights') #TODO we need to generate
        biases = tf.Variable(tf.zeros([h1_w_units]),
                            name='biases')
        hidden1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Varible(h2_w_units,
                              name='weights') #TODO we need to generate
        biases = tf.Variable(tf.zeros([h2_w_units]),
                            name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    prob_out = tf.matmul(hidden2, weights) + biases
    return prob_out

def loss(inputs, memory)
    #TBD
    return loss
def main(_):
  # Import data


  # Create the model
  x = tf.placeholder(tf.float32, [None, 32])

  # Define loss
  memory = tf.placeholder(tf.float32, [None, N*N*?])

  # Build the graph for the deep net
  prob_out = dqn_ttt(x)

  with tf.name_scope('loss'):
    = loss(inputs=prob_out,
           memory)
