import torchvision
import random

trainset = torchvision.datasets.MNIST(root='../../../data/MNIST', train=True, download=True)
testset = torchvision.datasets.MNIST(root='../../../data/MNIST', train=False, download=True)

datasets = {'train': trainset, 'test': testset}
def next_number(i, dataset, nr_digits):
    n = 0
    nr = list()
    for _ in range(nr_digits):
        x = next(i)
        _, c = dataset[x]
        n = n * 10 + c
        nr.append(str(x))
    return nr, n


def next_example(i, dataset, op, length):
    nr1, n1 = next_number(i, dataset, length)
    nr2, n2 = next_number(i, dataset, length)
    return nr1, nr2, op(n1, n2)


def generate_examples(dataset_name, op, length, out):
    dataset = datasets[dataset_name]
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    i = iter(indices)

    examples = list()
    while (True):
        try:
            examples.append(next_example(i, dataset, op, length))
        except StopIteration:
            break

    with open(out, 'w') as f:
        for example in examples:
            args1 = tuple('{}({})'.format(dataset_name, e) for e in example[0])
            args2 = tuple('{}({})'.format(dataset_name, e) for e in example[1])
            f.write('addition([{}], [{}], {}).\n'.format(','.join(args1), ','.join(args2), example[2]))


generate_examples('train', lambda x, y: x + y, 1, 'train.txt')
generate_examples('test', lambda x, y: x + y, 3, 'test.txt')
