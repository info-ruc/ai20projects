import tensorflow as tf

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist. load_data()

print("X_train.shape:",X_train.shape) print("X_test.shape: ",X_test.shape)

first_img = X_train[0]

# uncomment this line to see the pixel values
#print(first_img)

import matplotlib.pyplot as plt plt.imshow(first_img, cmap='gray') plt.show()
