import tensorflow as tf
def build_generator(img_shape, z_dim):
	model = tf.keras.models.Sequential()
    # Fully connected layer
	model.add(tf.keras.layers.Dense(128, input_dim=z_dim))
    # Leaky ReLU activation
	model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    # Output layer with tanh activation
	model.add(tf.keras.layers.Dense(28 * 28 * 1, activation='tanh'))
    # Reshape the Generator output to image dimensions
	model.add(tf.keras.layers.Reshape(img_shape))
	return model
def build_discriminator(img_shape):
	model = tf.keras.models.Sequential()
    # Flatten the input image
	model.add(tf.keras.layers.Flatten(input_shape=img_shape))
    # Fully connected layer
	model.add(tf.keras.layers.Dense(128))
    # Leaky ReLU activation
	model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
    # Output layer with sigmoid activation
	model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
	return model
def build_gan(generator, discriminator):
	# ensure that the discriminator is not trainable
	discriminator.trainable = False
	# the GAN connects the generator and descriminator
	gan = tf.keras.models.Sequential()
	# start with the generator:
	gan.add(generator)
	# then add the discriminator:
	gan.add(discriminator)
	# compile gan
	opt = tf.keras.optimizers.Adam(lr=0.0002,beta_1=0.5)
	gan.compile(loss='binary_crossentropy',optimizer=opt) 
	return gan
gen = build_generator(...)
dis = build_discriminator(...)
gan = build_gan(gen, dis)