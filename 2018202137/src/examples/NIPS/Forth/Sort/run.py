import sys

sys.path.append("../../../../")


from train import train_model
from model import Model
from network import Network
from standard_networks import FC
from data_loader import load
from optimizer import Optimizer
import torch

train = 2
test = 8

train_queries = load('data/train{}_test{}_train.txt'.format(train,test))
test_queries = load('data/train{}_test{}_test.txt'.format(train,test))


def neural_pred(network,i1,i2):
    d = torch.zeros(20)
    d[int(i1)] = 1.0
    d[int(i2)+10] = 1.0
    d = torch.autograd.Variable(d.unsqueeze(0))
    output = network.net(d)
    return output.squeeze(0)


fc1 = FC(20,2)
adam = torch.optim.Adam(fc1.parameters(), lr=1.0)
swap_net = Network(fc1, 'swap_net', neural_pred, optimizer=adam)


#with open('compare.pl') as f:
with open('quicksort.pl') as f:
    problog_string = f.read()

model = Model(problog_string, [swap_net])
optimizer = Optimizer(model, 32)

train_model(model, train_queries, 20, optimizer, test_iter=len(train_queries), test=lambda x:Model.accuracy(x, test_queries, test=True))