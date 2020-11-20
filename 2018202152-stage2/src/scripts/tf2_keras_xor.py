import tensorflow as tf
import numpy as np

# Logical XOR operator and "truth" values:
x = np.array([[0., 0.],[0., 1.],[1., 0.],[1.,1.]])
y = np.array([[0.], [1.], [1.], [0.]])

model = tf.keras.models.Sequential() 
model.add(tf.keras.layers.Dense(2, input_dim=2,
activation='relu')) 
model.add(tf.keras.layers.Dense(1))
print("compiling model...") 

model.compile(loss='mean_squared_error',
optimizer='adam')
print("fitting model...") 
model.fit(x,y,verbose=0,epochs=1000) 
pred = model.predict(x)

# Test final prediction print("Testing XOR operator") 
p1 = np.array([[0., 0.]])
p2 = np.array([[0., 1.]])
p3 = np.array([[1., 0.]])
p4 = np.array([[1., 1.]])

print(p1,":", model.predict(p1))
print(p2,":", model.predict(p2))
print(p3,":", model.predict(p3))
print(p4,":", model.predict(p4))