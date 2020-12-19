import argparse
import torchvision
from models.networks import MNIST_baseline_net, MNIST_Net
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torch.autograd import Variable
import os
from utils.logger import Logger
from utils.optimizer import myoptim
from data import load
from logic import Network
import time
from logic.model import Model
import sys
import math
import signal
import random
from logic.gradient_semiring import term2list2
from models import FC, RNN
from utils.optimizer import SGD


using_MNIST = [
    "MNIST_baseline",
    "MNIST_single_digit",
    "MNIST_multi_digit",
    "CoinUrn"
]

# used in MNIST baseline
def test_MNIST_baseline(net,test_dataset):
    confusion = np.zeros((19, 19), dtype=np.uint32)  # First index actual, second index predicted
    correct = 0
    n = 0
    N = len(test_dataset)
    for d, l in test_dataset:
        d = Variable(d.unsqueeze(0))
        if using_gpu:
            d = d.cuda()
        outputs = net.forward(d)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    print(confusion)
    F1 = 0
    for nr in range(19):
        TP = confusion[nr, nr]
        FP = sum(confusion[:, nr]) - TP
        FN = sum(confusion[nr, :]) - TP
        F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
    print('F1: ', F1)
    print('Accuracy: ', acc)
    return F1

# used in MNIST baseline to generate data
class MNIST_baseline_dataset(Dataset):

    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, l = self.data[index]
        return torch.cat((self.dataset[i1][0], self.dataset[i2][0]), 1), l


interrupt = False
zero_probability = False


def signal_handler(sig, frame):
        global interrupt
        print("Interrupted!")
        interrupt = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)



def train(model, optimizer, query, eps=1e-8): # 名为train,实为计算loss的一个函数
    loss = 0
    pos = True
    if type(query) is tuple:
        pos = query[1]
        query = query[0]
    ground = model.solve(query) # solve是怎么solve的需要搞清楚
    for k in ground:
        if k == query:
            p,d = ground[k]
            break
    if pos:
        if p <= 0:
            print('zero probability query:', query, p)
        loss_grad = -1.0/(p+eps)
        loss += -math.log(p+eps)
    else:
        loss_grad = 1.0/(1.0-p+eps)
        loss += -math.log(1-p+eps)
    for k, v in d.items():
        if type(k[0]) is str:
            name = k[0]
            i = tuple(term2list2(k[1]))
            grad = torch.FloatTensor(loss_grad*v)
            optimizer.backward(name,i,grad)
        else:
            optimizer.add_param_grad(k,loss_grad*float(v))

    return loss



def train_model(model,queries,nr_epochs,optimizer, loss_function = train, test_iter=1000,test=None,log_iter=100,snapshot_iter=None,snapshot_name='model',shuffle=True):
    signal.signal(signal.SIGINT, signal_handler) # 设置信号处理器，和C里的是相通的,感觉这里的设置会导致没有办法后台运行？
    i = 1
    accumulated_loss = 0
    logger = Logger()
    start = time.time()
    print("Training for {} epochs ({} iterations).".format(nr_epochs,nr_epochs*len(queries)))
    if test is not None:
        logger.log_list(i,test(model)) # 记录一下
    for epoch in range(nr_epochs):
        epoch_start = time.time()
        if interrupt:
            break
        print("Epoch",epoch+1)
        q_indices = list(range(len(queries)))
        if shuffle:
            random.shuffle(q_indices) # 这里打乱了有什么意义吗？ 下面还是把所有的数据用到了
        for q in q_indices:
            q = queries[q]
            iter_time = time.time()
            if interrupt:
                break
            loss = loss_function(model, optimizer, q) # 这里的loss_function 是上面传进来的参数
            accumulated_loss += loss
            optimizer.step()
            if snapshot_iter and i % snapshot_iter == 0:
                fname = '{}_iter_{}.mdl'.format(snapshot_name,i)
                print('Writing snapshot to '+fname)
                model.save_state(fname)
            if i % log_iter == 0:
                print('Iteration: ',i,'\tAverage Loss: ',accumulated_loss/log_iter)
                logger.log('time',i,iter_time - start)
                logger.log('loss',i,accumulated_loss/log_iter)
                for k in model.parameters:
                    logger.log(str(k),i,model.parameters[k])
                accumulated_loss = 0
            if test is not None and i % test_iter == 0:
                logger.log_list(i,test(model))
            i += 1
            sys.stdout.flush()
        optimizer.step_epoch()
        print('Epoch time: ',time.time()-epoch_start)
        sys.stdout.flush()
    return logger


# used in mnist single and multi digit
def neural_predicate(network, i):
    dataset = str(i.functor)
    i = int(i.args[0])
    if dataset == 'train':
        d, l = mnist_trainset[i]
    elif dataset == 'test':
        d, l = mnist_testset[i]
    d = Variable(d.unsqueeze(0))
    if using_gpu:
        d = d.cuda()

    output = network.net(d)
    return output.squeeze(0).cpu()

# used in forth add
def neural_predict_forth_add(network, i1, i2, carry):
    d = torch.zeros(30)
    d[int(i1)] = 1.0
    d[int(i2)+10] = 1.0
    d[int(carry)+20] = 1.0
    d = torch.autograd.Variable(d.unsqueeze(0))
    if using_gpu:
        d = d.cuda()
    outputs = network.net(d)
    return outputs.squeeze(0).cpu()

# used in forth sort
def neural_predict_forth_sort(network,i1,i2):
    d = torch.zeros(20)
    d[int(i1)] = 1.0
    d[int(i2)+10] = 1.0
    d = torch.autograd.Variable(d.unsqueeze(0))
    if using_gpu:
        d = d.cuda()
    output = network.net(d)
    return output.squeeze(0).cpu()

# used in coinurn  
def neural_predict_coinurn(network, i):
    dataset = str(i.functor)
    d, l = mnist_trainset[int(i)]
    # d = Variable(d.unsqueeze(0))
    d = Variable(d.unsqueeze(0))
    if using_gpu:
        d = d.cuda()
    output = network.net(d)
    return output.squeeze(0).cpu()

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

# used in forth WAP
def np1(net,sentence):
    if net.last[0] == str(sentence): #Caching
        return net.last[1]
    tokenized,numbers,indices = tokenize(str(sentence).strip('"'))
    data = torch.zeros(len(tokenized),1,len(vocab))
    for i,t in enumerate(tokenized):
        data[i,0,t] = 1.0
    data = Variable(data)
    if using_gpu:
        data = data.cuda()
    outputs = net.net(data,*indices)
    net.last = (str(sentence),outputs)
    return outputs.cpu()


# used in forth WAP
def np2(net, id):
    representation = np1(networks[0], id)
    if using_gpu:
        representation = representation.cuda()
    outputs = net.net(representation)
    return outputs.squeeze(0).cpu()

# used in coinurn
def colour_predicate(net, *colour):
    d = torch.FloatTensor(colour)
    d = Variable(d.unsqueeze(0))
    if using_gpu:
        d = d.cuda()
    outputs = net.net(d)
    return outputs.squeeze(0).cpu()


if "__main__" == __name__:
    # print("start")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", 
        help = ''' Please choose from
            "MNIST_baseline",
            "MNIST_single_digit",
            "MNIST_multi_digit",
            "Forth_Add",
            "Forth_Sort",
            "Forth_WAP",
            "CoinUrn"
            ''',
        type=str,
        default="MNIST_baseline",
        choices= {
            "MNIST_baseline",
            "MNIST_single_digit",
            "MNIST_multi_digit",
            "Forth_Add",
            "Forth_Sort",
            "Forth_WAP",
            "CoinUrn"
        }
    )
    parser.add_argument("--gpu", type=str, help="Please select a gpu, otherwise gpu will be disabled")
    args = parser.parse_args()
    selection = args.selection
    if selection == None:
        print("Please select an experiment")
        exit(1)

    using_gpu = False
    if args.gpu != None:
        os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu
        print("====== Using gpu {} ======".format(args.gpu))
        using_gpu = True
    else:
        print("====== gpu disabled ======")

    print("====== Selected experiment {} ======".format(selection))
    if selection in using_MNIST:
        print("====== Loading MNIST ======", end=' ')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081, ))
        ])  
        mnist_trainset = torchvision.datasets.MNIST(root='data/MNIST', train=True, download=True, transform=transform)
        mnist_testset = torchvision.datasets.MNIST(root='data/MNIST', train=False, download=True, transform=transform)
        print("Done.")

    '''
        baseline based on MNIST 

        cat two MNIST images together and predict on it
    '''
    if "MNIST_baseline" == selection:
        net = MNIST_baseline_net()
        if using_gpu:
            net = net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        criterion = nn.NLLLoss()
        train_dataset = MNIST_baseline_dataset(mnist_trainset, 'data/MNIST/baseline_train_data.txt')
        test_dataset = MNIST_baseline_dataset(mnist_testset, 'data/MNIST/baseline_test_data.txt')
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)


        iters = 1
        test_period = 500
        log_period = 50
        running_loss = 0.0
        log = Logger()
        epoches = 1

        for epoch in range(epoches):
            for data in trainloader:
                inputs, labels = data
                if using_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.data.item()
                if iters % log_period == 0:
                    print('Iteration: ', iters * 2, '\tAverage Loss: ', running_loss / log_period)
                    log.log('loss', iters * 2, running_loss / log_period)
                    running_loss = 0
                if iters % test_period == 0:
                    log.log('F1', iters * 2, test_MNIST_baseline(net, test_dataset))
                iters += 1

    
    if "MNIST_single_digit" == selection:
        queries = load('data/MNIST/single_digit_train.txt')
        test_queries = load('data/MNIST/single_digit_test.txt')

        with open('problog/addition.pl') as f:
            problog_string = f.read()
        # network = MNIST_Net()
        if using_gpu:
            network = MNIST_Net().cuda()
        else:
            network = MNIST_Net()
        net = Network(network, 'mnist_net', neural_predicate)
        net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        model = Model(problog_string, [net], caching=False)

        optimizer = myoptim(model, 2)

        train_model(model, queries, 1, optimizer, test_iter=1000, test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)


    if "MNIST_multi_digit" == selection:
        train_queries = load('data/MNIST/multi_digit_train.txt')
        test_queries = load('data/MNIST/multi_digit_test.txt')[:100]

        with open('problog/multi_digit.pl') as f:
            problog_string = f.read()

        if using_gpu:
            network = MNIST_Net().cuda()
        else:
            network = MNIST_Net()
        net = Network(network, 'mnist_net', neural_predicate)
        net.optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        model = Model(problog_string, [net], caching=False)
        optimizer = myoptim(model, 2)

        train_model(model, train_queries, 1, optimizer, test_iter=1000, test=lambda x: x.accuracy(test_queries, test=True), snapshot_iter=10000)



    if "Forth_Add" == selection:
        train = 2
        test = 8

        train_queries = load('data/Forth/Add/train{}_test{}_train.txt'.format(train,test))
        test_queries = load('data/Forth/Add/train{}_test{}_dev.txt'.format(train,test))

        with open('problog/choose.pl') as f:
            problog_string = f.read()
        

        net1 = FC(30,20, 10)
        if using_gpu:
            net1 = net1.cuda()
        network1 = Network(net1, 'neural1', neural_predict_forth_add)
        network1.optimizer = torch.optim.Adam(net1.parameters(), lr=0.05)

        net2 = FC(30,15 ,2)
        if using_gpu:
            net2 = net2.cuda()
        network2 = Network(net2, 'neural2', neural_predict_forth_add)
        network2.optimizer = torch.optim.Adam(net2.parameters(), lr=0.05)

        model = Model(problog_string,[network1,network2], caching=False)
        optimizer = myoptim(model, 32)
        logger = train_model(model, train_queries, 40, optimizer, test_iter=len(train_queries)*4, test=lambda x: x.accuracy(test_queries, test=True))

    if "Forth_Sort" == selection:

        train = 2
        test = 8

        train_queries = load('data/Forth/Sort/train{}_test{}_train.txt'.format(train,test))
        test_queries = load('data/Forth/Sort/train{}_test{}_test.txt'.format(train,test))

        fc1 = FC(20,2)
        if using_gpu:
            fc1 = fc1.cuda()
        adam = torch.optim.Adam(fc1.parameters(), lr=1.0)
        swap_net = Network(fc1, 'swap_net', neural_predict_forth_sort, optimizer=adam)


        #with open('compare.pl') as f:
        with open('problog/quicksort.pl') as f:
            problog_string = f.read()

        model = Model(problog_string, [swap_net])
        optimizer = myoptim(model, 32)

        train_model(model, train_queries, 20, optimizer, test_iter=len(train_queries), test=lambda x:Model.accuracy(x, test_queries, test=True))


    if "Forth_WAP" == selection:

        vocab = dict()

        with open('data/Forth/WAP/vocab_746.txt') as f:
            for i,word in enumerate(f):
                word = word.strip()
                vocab[i] = word
                vocab[word] = i

        rnn = RNN(len(vocab),75)
        network1 = FC(600,6)
        network2 = FC(600,4)
        network3 = FC(600,2)
        network4 = FC(600,4)
        if using_gpu:
            rnn = rnn.cuda()
            network1 = network1.cuda()
            network2 = network2.cuda()
            network3 = network3.cuda()
            network4 = network4.cuda()


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


        train_queries = load('data/Forth/WAP/train.txt')
        test_queries = load('data/Forth/WAP/dev.txt')

        with open('problog/wap.pl') as f:
            problog_string = f.read()

        model = Model(problog_string, networks, caching=True)
        optimizer = myoptim(model, 50)


        train_model(model, train_queries, 40, optimizer, log_iter=150, test_iter=150, test=lambda x: Model.accuracy(x, test_queries, test=True))


    if "CoinUrn" == selection:

        coin_network = MNIST_Net(2)
        if using_gpu:
            coin_network = coin_network.cuda()
        coin_net = Network(coin_network, 'coin_net', neural_predict_coinurn)
        coin_net.optimizer = optim.Adam(coin_network.parameters(), lr=1e-3)

        # 第3个参数为function

        colour_network = FC(3, 3)
        if using_gpu:
            colour_network = colour_network.cuda()
        colour_net = Network(colour_network, 'colour_net', colour_predicate)
        colour_net.optimizer = optim.Adam(colour_network.parameters(), lr=1.0)


        queries = load('data/CoinUrn/train.txt')
        test_queries = load('data/CoinUrn/test.txt')

        with open('problog/model.pl') as f:
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
