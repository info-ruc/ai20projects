import numpy as np
trainx=np.linspace(-1,1,11)
trainy=4*trainx+np.random.randn(*trainx.shape)*0.5

print("trainx:",trainx)
print("trainy:",trainy)