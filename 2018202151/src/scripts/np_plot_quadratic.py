import numpy as np
import matplotlib.pyplot as plt

#see what happens with this set if values:

x=np.linspace(-5,5,num=100)[:,None]
y = -0.5 + 2.2*x +0.3*x**2 + 2*np.random.randn(100,1)

#print("x:",x)
plt.plot(x,y) 
plt.show()
