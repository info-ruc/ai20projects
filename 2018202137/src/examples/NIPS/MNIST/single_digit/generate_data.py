import torchvision
import random

trainset = torchvision.datasets.MNIST(root='../../../data/MNIST', train=True, download=True)
testset = torchvision.datasets.MNIST(root='../../../data/MNIST', train=False, download=True)

datasets = {'train': trainset, 'test': testset}


def next_example(dataset, i):
    x, y = next(i), next(i)
    (_, c1), (_, c2) = dataset[x], dataset[y]
    return x, y, c1 + c2


def gather_examples(dataset_name, filename):
    dataset = datasets[dataset_name]
    examples = list()
    i = list(range(len(dataset)))
    random.shuffle(i)
    i = iter(i)
    while True:
        try:
            examples.append(next_example(dataset, i))
        except StopIteration:
            break

    with open(filename, 'w') as f:
        for example in examples:
            args = tuple('{}({})'.format(dataset_name, e) for e in example[:-1])
            f.write('addition({},{},{}).\n'.format(*args, example[-1]))


gather_examples('train', 'train_data.txt')
gather_examples('test', 'test_data.txt')
