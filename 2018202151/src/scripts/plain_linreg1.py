import numpy as np
import matplotlib.pyplot as plt

X = [0,0.12,0.25,0.27,0.38,0.42,0.44,0.55,0.92,1.0]
Y = [0,0.15,0.54,0.51,
0.34,0.1,0.19,0.53,1.0,0.58]
costs = []

#Step 1: Parameter initialization W = 0.45
b = 0.75
W = 0.45
for i in range(1, 100): 
    #Step 2: Calculate Cost
    Y_pred = np.multiply(W, X) + b 
    Loss_error = 0.5 * (Y_pred - Y)**2 
    cost = np.sum(Loss_error)/10
    #Step 3: Calculate dW and db 
    db = np.sum((Y_pred - Y))
    dw = np.dot((Y_pred - Y), X) 
    costs.append(cost)
    #Step 4: Update parameters:
    W = W - 0.01*dw
    b = b - 0.01*db

    if i%10 == 0:
        print("Cost at", i,"iteration = ", cost)

    #Step 5: Repeat via a for loop with 1000 iterations 
    #Plot cost versus # of iterations
    print("W = ", W,"& b = ",	b) 
    plt.plot(costs) 
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)') 
    plt.show()
