import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from network import Network
from examples.NIPS.MNIST.mnist import MNIST_Net, neural_predicate


class FC(nn.Module):
    def __init__(self, in_size, out_size):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size,bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.softmax(x)
        return x


def colour_predicate(net, *colour):
    d = torch.FloatTensor(colour)
    d = Variable(d.unsqueeze(0))
    outputs = net.net(d)
    return outputs.squeeze(0)


coin_network = MNIST_Net(2)
coin_net = Network(coin_network, 'coin_net', neural_predicate)
coin_net.optimizer = optim.Adam(coin_network.parameters(), lr=1e-3)

colour_network = FC(3, 3)
colour_net = Network(colour_network, 'colour_net', colour_predicate)
colour_net.optimizer = optim.Adam(colour_network.parameters(), lr=1.0)
