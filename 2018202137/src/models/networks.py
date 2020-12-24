import torch.nn as nn
import torch

class MNIST_baseline_net(nn.Module):
    def __init__(self):
        super(MNIST_baseline_net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(

            nn.Linear(16 * 11 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 19),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 11 * 4)
        x = self.classifier(x)
        return x


class MNIST_Net(nn.Module):
    def __init__(self, N=10):
        super(MNIST_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2), # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5), # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2), # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier =  nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x



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


class RNN(nn.Module):
    def __init__(self,vocab_size,hidden_size):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(vocab_size,hidden_size,1,bidirectional=True)
        # print(vocab_size)
        # exit(1)

    def forward(self, x,n1,n2,n3):
        x ,_ = self.lstm(x)
        # print(x.shape)
        x = torch.cat((x[-1,...],x[n1,...],x[n2,...],x[n3,...]),1)
        x.view(1,-1)
        # print(x.shape)
        return x



