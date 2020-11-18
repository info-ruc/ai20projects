#Simple RNN and MNIST dataset
import tensorflow as tf
import numpy as np

# instantiate mnist and load data:
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# one-hot encoding for all labels to create 1x10
# vectors that are compared with the final layer:
y_train = tf.keras.utils.to_categorical(y_train)
y_test  = tf.keras.utils.to_categorical(y_test)

#resize and normalize the 28x28 images:
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size])
x_test  = np.reshape(x_test, [-1, image_size, image_size])
x_train = x_train.astype('float32') / 255
x_test  = x_test.astype('float32')  / 255

#initialize some hyper-parameters:
input_shape = (image_Size, image_size)
batch_Size = 128
hidden_units = 128
dropout_rate = 0.3

#RNN-based Keras model with 128 hidden units:
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(units=hidden_units,
                                   dropout=dropout_rate,
                                   input_shape=input_shape))
model.add(tf.keras.layers.Dense(num_labels))
model.add(tf.keras.layers.Actication('softmax'))
model.summary()

model.compile(loss='categotical_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])

#train the network on the training data:
model.fit(x_train, y_train, epochs=8, batch_size=batch_Size)

#calculate and then display the accuracy:
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))