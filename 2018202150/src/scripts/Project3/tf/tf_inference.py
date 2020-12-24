import tensorflow as tf
INPUT_NODE=784
OUTPUT_NODE=43     

IMAGE_SIZE=28
NUM_CHANNELS = 1
NUM_LABELS=64            #nodes of second full connect layer

#1st convolution layer
CONV1_DEEP=32
CONV1_SIZE=5
#2nd convolution layer。
CONV2_DEEP=64
CONV2_SIZE=5
#nodes of full connect
FC_SIZE=512
def inference(input_tensor,train,regularizer):
    #Declare the variables of the first convolutional layer and implement the propagation process.
    
    #Separate variables at different levels by using different namespaces.
    with tf.variable_scope('layer1-conv1'):
        conv1_weights=tf.get_variable(
            "weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.01))
        #use a filter of size 5x5 and depth 32, strip=1
        conv1=tf.nn.conv2d(
            input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME'
        )
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    # the forward propagation process of the second pooling layer. pooling = 2, using all zero padding, strip = 2.
    # input of this layer is the output of the last layer, as 28 x 28 x 32 matrix, the output is a 14 x 14 x 32 matrix
    with tf.name_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(
            relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'
        )
    #Declare the variables of the third convolutional layer and implement the forward propagation process.
    # the input is 14x14x32,output is 14x14x64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights=tf.get_variable("conv2_weights",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        #filter with width=5,depth=64,strip=1,padding with all zero
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #the 4th layer pooling layer is the same as the 2nd layer
    with tf.name_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #The output of the fourth layer is transformed into the input format of the fully connected layer and straightened into a vector
    pool_shape=pool2.get_shape().as_list()

    #pool_shape[0] is the number of data in a batch
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]

    #tf.reshape()transform the output of fourth layer into a vector of batch size
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])
    #the 5th layer, full connection.
    # input is a vector of size 3136, output is a vector of size 512
    # use drop out
    with tf.variable_scope('layer5-fc1'):
        fc1_weights=tf.get_variable(
            "weight",[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        #normalization
        if regularizer !=None :
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable(
            "bias",[FC_SIZE],initializer=tf.constant_initializer(0.1))

        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train: fc1=tf.nn.dropout(fc1,0.5)

    #the 6th layer, full connection.
    # output is a vector of size 64
    with tf.variable_scope('layer6-fc2'):
        fc2_weights=tf.get_variable("weight",[FC_SIZE,NUM_LABELS],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.get_variable(
            "bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases
    return logit