from train import train_model
from examples.NIPS.CoinUrn.nets import colour_net, coin_net
from data_loader import load
from optimizer import SGD
from model import Model

queries = load('train.txt')
test_queries = load('train.txt')

with open('model.pl') as f:
    problog_string = f.read()

def lr(self):
    if self.epoch < 2:
        return 1e-4
    elif self.epoch < 4:
        return 1e-3
    elif self.epoch < 8:
        return 1e-2
    return 1e-3

model = Model(problog_string, [colour_net, coin_net], caching=False)
optimizer = SGD(model, 16)
SGD.get_lr = lr
train_model(model, queries, 10, optimizer,test_iter=256, test = lambda x: Model.accuracy(x, test_queries), snapshot_iter=10000)
