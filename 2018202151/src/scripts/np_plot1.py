import numpy as np
import matplotlib.pyplot as plt

x=np.random.randn(15,1)
y=2.5*x+5+0.2*np.random.randn(15,1)

print("x:",x)
print("y:",y)

plt.scatter(x,y)
plt.show()