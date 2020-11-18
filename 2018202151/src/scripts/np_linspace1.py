import numpy as np

trainX = np.linspace(-1, 1, 6)
trainY = 3*trainX+ np.random.randn(*trainX. shape)*0.5

print("trainX: ", trainX)
print("trainY: ", trainY)