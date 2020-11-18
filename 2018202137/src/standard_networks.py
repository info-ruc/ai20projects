import torch.nn as nn


class FC(nn.Module):

    def __init__(self, *sizes):
        super(FC, self).__init__()
        layers = []
        for i in range(len(sizes)-2):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        layers.append(nn.Softmax(1))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
