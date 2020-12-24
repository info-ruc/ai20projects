import tensorflow as tf

timesteps = 30
input_dim = 12

# number of units in RNN cell
units = 512

#number of classes to be identified
n_classes = 5

#Keras Sequential model with RNN and Dense layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(units=units,dropout=0.2,
                                    input_shape=(timesteps,input_dim)))
model.add(tf.keras.layers.Dense(n_classes,
                               activation='softmax'))

#model loss function and optimizer
model.compile(loss='categorical_crossentropy',
             optimizer=tf.keras.optimizers.Adam(),
             metrics=['accuracy'])

model.summary()