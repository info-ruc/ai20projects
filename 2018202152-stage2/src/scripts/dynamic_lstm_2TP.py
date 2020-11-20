import tensorflow as tf
import numpy as np
n_steps = 2 # number of time steps
n_inputs = 3 # number of inputs per time unit
n_neurons = 5 # number of hidden units
X_batch = np.array([
 # t = 0      t = 1
 [[0, 1, 2], [9, 8, 7]], # instance 0
 [[3, 4, 5], [0, 0, 0]], # instance 1
 [[6, 7, 8], [6, 5, 4]], # instance 2
 [[9, 0, 1], [3, 2, 1]], # instance 3
])
seq_length_batch = np.array([2, 1, 2, 2])
X = tf.placeholder(dtype=tf.float32,shape=[None,n_steps,n_inputs])
seq_length = tf.placeholder(tf.int32, [None])
basic_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, 
                                    sequence_length=seq_length, dtype=tf.float32)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	outputs_val, states_val = sess.run([outputs, states],
	feed_dict={X:X_batch, seq_length:seq_length_batch})
	print("X_batch shape:", X_batch.shape) # (4,2,3)
	print("outputs_val shape:", outputs_val.shape) # (4,2,5)
 	print("states: ", states_val) # LSTMStateTuple(...)
 	print("outputs_val:",outputs_val)
 	print("----------------------------\n")
	print("states_val: ",states_val)