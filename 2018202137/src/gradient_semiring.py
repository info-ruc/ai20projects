from problog.evaluator import Semiring
from problog.logic import Constant, Term, is_variable
import numpy as np


class SemiringGradient(Semiring):

    def __init__(self, model, shape):
        Semiring.__init__(self)
        self.shape = shape
        self.model = model

    def zero(self):
        return 0.0, np.zeros(self.shape.length)

    def one(self):
        return 1.0, np.zeros(self.shape.length)

    def plus(self, a, b):
        return a[0]+b[0], a[1]+b[1]

    def times(self, a, b):
        return a[0]*b[0], b[0]*a[1]+a[0]*b[1]

    def value(self, a, key=None):
        if type(a) is Constant:
            return float(a), np.zeros(self.shape.length)
        elif type(a) is Term:
            if a.functor == 'nn':
                network = self.model.networks[str(a.args[0])]
                input = a.args[1]
                if len(a.args) == 3:
                    output = a.args[2]
                    p = network.get_probability(term2list2(input),output)
                    i = self.shape.get_index_network(str(a.args[0]), input)+self.shape.get_index_network_output(str(a.args[0]), output)
                elif len(a.args) == 2:
                    p = network.get_probability(term2list2(input))
                    i = self.shape.get_index_network(str(a.args[0]),input)
                diff = np.zeros(self.shape.length)
                diff[i] = 1.0
                return p, diff

            elif a.functor == 't':
                p = self.model.parameters[a.location]
                i = self.shape.get_index_parameter(a.location)
                diff = np.zeros(self.shape.length)
                diff[i] = 1.0
                for _, ad in self.model.ADs.items():
                    if a.location in ad:
                        for head in ad:
                            if not a.location == head:
                                j = self.shape.get_index_parameter(head)
                                diff[j] = -self.model.parameters[head]/(1.0-p)

                return p, diff
            else:
                raise ValueError('Bad functor: {} at {}'.format(a.functor, a.location))

    def negate(self, a):
        return 1.0-a[0], -1.0*a[1]

    def is_dsp(self):
        return True

    def is_one(self, a):
        return (1.0 - 1e-12 < a[0] < 1.0 + 1e-12) and (np.count_nonzero(a[1]) == 0)

    def is_zero(self, a):
        return (-1e-12 < a[0] < 1e-12) and (np.count_nonzero(a[1]) == 0)

    def normalize(self, a, z):
        diff = np.zeros(self.shape.length)
        for i in range(self.shape.length):
            diff[i] = (a[1][i]*z[0]-z[1][i]*a[0])/(z[0]**2)
        return a[0]/z[0], diff


def term2list2(term):
    result = []
    while not is_variable(term) and term.functor == '.' and term.arity == 2:
        result.append(term.args[0])
        term = term.args[1]
    if not term == Term('[]'):
        raise ValueError('Expected fixed list.')
    return result