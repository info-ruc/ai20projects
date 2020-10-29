import numpy as np

trainX = np.linspace(-1, 1, 11)
trainY = 4*trainX + np.random.randn(*trainX.shape)*0.5
print("trainX: ",trainX)
print("trainY: ",trainY)