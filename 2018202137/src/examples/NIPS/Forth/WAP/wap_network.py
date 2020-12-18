import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from network import Network
from standard_networks import FC

vocab = dict()

with open('vocab_746.txt') as f:
    for i,word in enumerate(f):
        word = word.strip()
        vocab[i] = word
        vocab[word] = i


def tokenize(sentence):
    sentence = sentence.split(' ')
    tokens = []
    numbers = list()
    indices = list()
    for i,word in enumerate(sentence):
        if word.isdigit():
            numbers.append(int(word))
            tokens.append('<NR>')
            indices.append(i)
        else:
            if word in vocab:
                tokens.append(word)
            else:
                tokens.append('<UNK>')
    return [vocab[token] for token in tokens],numbers,indices


class RNN(nn.Module):
    def __init__(self,vocab_size,hidden_size):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(vocab_size,hidden_size,1,bidirectional=True)

    def forward(self, x,n1,n2,n3):
        x ,_ = self.lstm(x)
        x = torch.cat((x[-1,...],x[n1,...],x[n2,...],x[n3,...]),1)
        x.view(1,-1)
        return x



rnn = RNN(len(vocab),75)
network1 = FC(600,6)
network2 = FC(600,4)
network3 = FC(600,2)
network4 = FC(600,4)


def np1(net,sentence):
    if net.last[0] == str(sentence): #Caching
        return net.last[1]
    tokenized,numbers,indices = tokenize(str(sentence).strip('"'))
    data = torch.zeros(len(tokenized),1,len(vocab))
    for i,t in enumerate(tokenized):
        data[i,0,t] = 1.0
    outputs = net.net(Variable(data),*indices)
    net.last = (str(sentence),outputs)
    return outputs



def np2(net, id):
    representation = np1(networks[0], id)
    outputs = net.net(representation)
    return outputs.squeeze(0)



networks = [Network(rnn, 'nn_rnn', np1),
            Network(network1, 'nn_permute', np2),
            Network(network2, 'nn_op1', np2),
            Network(network3, 'nn_swap', np2),
            Network(network4, 'nn_op2', np2)]
networks[0].last = ('',None)

networks[0].optimizer = optim.Adam(rnn.parameters(), lr=0.02)
networks[1].optimizer = optim.Adam(network1.parameters(), lr=0.02)
networks[2].optimizer = optim.Adam(network2.parameters(), lr=0.02)
networks[3].optimizer = optim.Adam(network3.parameters(), lr=0.02)
networks[4].optimizer = optim.Adam(network4.parameters(), lr=0.02)
