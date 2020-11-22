import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_ labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28,
1))
test_images	= test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values: from the range 0-255 to the range 0-1
train_images, test_images = train_images/255.0, test_images/255.0

model = tf.keras.models.Sequential() model.add(tf.keras.layers.Conv2D(32, (3, 3),
activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten()) model.add(tf.keras.layers.Dense(64,
activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.summary() model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy', metrics=[‘accuracy'])

model.fit(train_images, train_labels, epochs=1)
test_loss, test_acc = model.evaluate(test_images,
test_labels)
print(test_acc)

# predict the label of one image
test_image = np.expand_dims(test_images[300], axis = 0)
plt.imshow(test_image.reshape(28,28)) plt.show()

result = model.predict(test_image) print("result:", result) print("result.argmax():", result.argmax())
