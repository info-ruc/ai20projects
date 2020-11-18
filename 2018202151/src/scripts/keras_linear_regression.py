##############################################################
#Keep in mind the following important points: 
#1) Always standardize both input features and target variable:
#doing so only on input feature produces incorrect predictions
#2) Data might not be normally distributed: check the data and
#based on the distribution apply StandardScaler, MinMaxScaler,
#Normalizer or RobustScaler 
###############################################################

import tensorflow as tf 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
print("finish")

df = pd.read_csv('housing.csv') 
X	= df.iloc[:,0:13]
y	= df.iloc[:,13].values

mmsc = MinMaxScaler()
X	= mmsc.fit_transform(X)
y	= y.reshape(-1,1)
y	= mmsc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# this Python method creates a Keras model 
def build_keras_model():
    model = tf.keras.models.Sequential() 
    model.add(tf.keras.layers.Dense(units=13,input_dim=13))
    model.add(tf.keras.layers.Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae','accuracy'])
    return model

batch_size=32 
epochs = 40

# specify the Python method 'build_keras_model' to create a Keras model
# using the implementation of the scikit-learn regressor API for Keras
model = tf.keras.wrappers.scikit_learn.KerasRegressor(
    build_fn=build_keras_model,
    batch_size=batch_size,
    epochs=epochs)

# train ('fit') the model and then make predictions: 
model.fit(X_train, y_train) 
y_pred = model.predict(X_test) 
#print("y_test:",y_test) 
#print("y_pred:",y_pred)

# scatter plot of test values-vs-predictions
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred) 
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r*--') 
ax.set_xlabel('Calculated') 
ax.set_ylabel('Predictions') 
plt.show()
