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
test_queries = load('data/train{}_test{}_dev.txt'.format(train,test))

with open('choose.pl') as f:
    problog_string = f.read()

def neural_pred(network, i1, i2, carry):
    d = torch.zeros(30)
    d[int(i1)] = 1.0
    d[int(i2)+10] = 1.0
    d[int(carry)+20] = 1.0
    d = torch.autograd.Variable(d.unsqueeze(0))
    outputs = network.net(d)
    return outputs.squeeze(0)


net1 = FC(30, 25, 10)
network1 = Network(net1, 'neural1', neural_pred)
network1.optimizer = torch.optim.Adam(net1.parameters(), lr=0.05)

net2 = FC(30, 5, 2)
network2 = Network(net2, 'neural2', neural_pred)
network2.optimizer = torch.optim.Adam(net2.parameters(), lr=0.05)

model = Model(problog_string,[network1,network2], caching=False)
optimizer = Optimizer(model, 32)
logger = train_model(model, train_queries, 40, optimizer, test_iter=len(train_queries)*4, test=lambda x: x.accuracy(test_queries, test=True))
