import random
import torchvision
import colorsys

coin = {'heads':0.3,'tails':0.7}
urn1 = {'red':0.4,'blue':0.6}
urn2 = {'red':0.3,'green':0.3,'blue':0.4}

colors = {'red':(1.0,0.0,0.0),'green':(0.0,0.1,0.0),'blue':(0.0,0.0,1.0)}
color_scaling = 0.03




def random_color(color):
    h,s,v = colorsys.rgb_to_hsv(*colors[color])
    h = (random.gauss(h,color_scaling))%1.0
    s = max(0.0,min(1.0,random.gauss(s,color_scaling)))
    v = max(0.0,min(1.0,random.gauss(v,color_scaling)))
    r,g,b = colorsys.hsv_to_rgb(h,s,v)
    return r,g,b

def random_coin_image(coins,side):
    return random.choice(coins[side])


def split_dataset(dataset):
    coins = {'heads': list(), 'tails': list()}
    for i,(_,c) in enumerate(dataset):
        if c == 1 :
            coins['tails'].append(i)
        elif c == 0:
            coins['heads'].append(i)

    return coins

def generate_list(N,distrib):
    l = list()
    for k in distrib:
        n = int(N*distrib[k])
        l += [k]*n
    random.shuffle(l)
    return l
def generate_examples(data, N, fname):
    coins = generate_list(N,coin)
    balls1 = generate_list(N,urn1)
    balls2 = generate_list(N,urn2)
    examples = list(zip(coins,balls1,balls2))
    lines = list()
    for c,b1,b2 in examples:
        outcome = 'loss'
        if c == 'heads' and (b1 == 'red' or b2 == 'red'):
            outcome = 'win'
        elif b1 == b2:
            outcome = 'win'
        c =  str(random_coin_image(data,c))
        b1 = ','.join([str(col) for col in random_color(b1)])
        b2 = ','.join([str(col) for col in random_color(b2)])
        lines.append('game({},[{}],[{}],{}).'.format(c,b1,b2,outcome))

    with open(fname, 'w') as f:
        f.write('\n'.join(lines))

trainset = torchvision.datasets.MNIST(root='../../data/MNIST', train=True, download=True)
testset = torchvision.datasets.MNIST(root='../../data/MNIST', train=False, download=True)
train_coins = split_dataset(trainset)
test_coins = split_dataset(testset)

generate_examples(train_coins,256,'train.txt')
generate_examples(train_coins,64,'test.txt')
