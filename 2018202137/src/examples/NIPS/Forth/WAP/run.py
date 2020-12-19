import sys

sys.path.append("../../../../")

from train import train_model
from examples.NIPS.Forth.WAP.wap_network import networks
from model import Model
from data_loader import load
from optimizer import Optimizer
from random import shuffle

train_queries = load('train.txt')
test_queries = load('dev.txt')

with open('wap.pl') as f:
    problog_string = f.read()

model = Model(problog_string, networks, caching=True)
optimizer = Optimizer(model, 50)


train_model(model, train_queries, 40, optimizer, log_iter=150, test_iter=150, test=lambda x: Model.accuracy(x, test_queries, test=True))
