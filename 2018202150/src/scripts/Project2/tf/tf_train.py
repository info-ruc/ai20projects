import tensorflow as tf
from tf_inference import inference
import numpy as np
import os

INPUT_NODE=784
OUTPUT_NODE=43     
IMAGE_SIZE=28
NUM_CHANNELS = 1
NUM_LABELS=64  
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARZATION_RATE = 0.0001
TRAINING_STEPS = 30001
MOVING_AVERAGE_DECARY = 0.99  # Moving average decay rate
MODEL_SAVE_PATH='E:AutoDrive/mymodel/'
MODEL_NAME = "model.ckpt"

def train(xs,ys):
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,  # batch size
                        IMAGE_SIZE,  # size of image
                        IMAGE_SIZE,
                        NUM_CHANNELS],  # depth of image
                       name='x-input'
                       )
    y_ = tf.placeholder(tf.float32, [None,OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)
    y =inference(x, True, regularizer)          #predict using inference
    global_step = tf.Variable(0, trainable=False)

    # loss function,moving average rate,learning rate
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECARY, global_step)
    variable_average_op = variable_average.apply(
        tf.trainable_variables()
    )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        400,
        LEARNING_RATE_DECAY,
        staircase = True
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)#最小化loss来进行反向传播
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # initialize the saver class
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        a=len(xs)
        for i in range(TRAINING_STEPS):
            start=(i*BATCH_SIZE)%a
            end=min(start+BATCH_SIZE,a)
            xs_batch=xs[start:end]
            ys_batch=ys[start:end]
            reshaped_xs = np.reshape(xs_batch, (BATCH_SIZE,
                                          IMAGE_SIZE,
                                          IMAGE_SIZE,
                                          NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],feed_dict={x:reshaped_xs,y_:ys_batch.eval()})
            #Save every 1000 training sessions
            if i % 100== 0:
                print("After %d training step(s),loss on training " "batch is %g." % (step, loss_value))
                saver.save(
                    sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step
                )