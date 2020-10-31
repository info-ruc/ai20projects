import sys

sys.path.append("../../../../")

from train import train_model
from data_loader import load
from examples.NIPS.MNIST.mnist import test_MNIST, MNIST_Net, neural_predicate
from model import Model
from optimizer import Optimizer
from network import Network
import torch

queries = load('train_data.txt')
test_queries = load('test_data.txt')



print("Load data succeeded!")
sys.stdout.flush()

with open('addition.pl') as f:
    problog_string = f.read()

    '''
    这个变量 problog string是这个样子的

    nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

    addition(X,Y,Z) :- digit(X,X2), digit(Y,Y2), Z is X2+Y2.

    第二句能看懂，第一句可能是用网络形成一个谓词

    '''

network = MNIST_Net()
net = Network(network, 'mnist_net', neural_predicate)
net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
model = Model(problog_string, [net], caching=False)
optimizer = Optimizer(model, 2)

train_model(model, queries, 1, optimizer, test_iter=1000, test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)
