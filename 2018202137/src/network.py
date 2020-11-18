import torch
import problog
from problog.extern import problog_export

from logic import term2list2


def count_backward(self, x, y):
    self.n += 1



class Network(object):
    def __init__(self, net, name, function, optimizer=None, model=None):
        self.evaluated = dict()
        self.net = net
        self.net.n = 0
        self.net.register_backward_hook(count_backward)
        self.name = name
        self.function = function
        self.optimizer = optimizer
        self.model = model

    def normalize(self):
        for param in self.net.parameters():
            if param.grad is not None:
                param.grad /= self.net.n
        self.net.n = 0

    def backward(self, i, grad):
        if self.optimizer is not None:
            self.evaluated[i].backward(grad, retain_graph=True)

    def register_external(self, *args, **kwargs):
        pass
        
    def clear(self):
        self.evaluated.clear()

    def step(self):
        if self.optimizer is not None:
            self.normalize()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def load(self, location):
        self.net.load_state_dict(torch.load(location))

    def save(self, location):
        torch.save(self.net.state_dict(), location)

    def instantiate(self, term):
        if type(term) is problog.logic.Clause:
            annotation = term.head.probability.args
        else:
            annotation = term.probability.args
        if len(annotation) == 4:
            return AD_Network(self.net, self.name, self.function, self.optimizer, self.model, term)
        elif len(annotation) == 3:
            return Det_Network(self.net, self.name, self.function, self.optimizer, self.model,term)
        elif len(annotation) == 2:
            return Fact_Network(self.net, self.name, self.function, self.optimizer, self.model,term)

    def evaluate(self, inputs):
        probs = self.function(self, *inputs)
        self.evaluated[tuple(inputs)] = probs


class AD_Network(Network):
    def __init__(self, net, name, function, optimizer, model, term):
        Network.__init__(self, net, name, function, optimizer, model)
        if type(term) is problog.logic.Clause:
            head, body = term.head, term.body
        else:
            head, body = term, None
        self.input_terms = term2list2(head.probability.args[1])
        self.output_variable = head.probability.args[2]
        self.output_domain = term2list2(head.probability.args[3])
        subst = {var:var for var in self.input_terms}
        head.probability = head.probability.with_args(*head.probability.args[:-1])
        heads = []
        for sub in self.output_domain:
            subst[self.output_variable] = sub
            heads.append(head.apply(subst))
        self.term = problog.logic.AnnotatedDisjunction(heads,body)

        extra_term = problog.logic.Term(head.probability.args[0], *self.input_terms, self.output_variable)
        self.test_term = head << (extra_term & body if body else extra_term)

    def test_predicate(self, *inputs):
        output = self.function(self,*inputs)
        _, predicted = torch.max(output.data, 0)
        return self.output_domain[predicted]

    def register_external(self, _, test_model):
        problog_export.database = test_model
        signature = ['+term'] * len(self.input_terms) + ['-term']
        problog_export(*signature)(self.test_predicate, funcname=self.name, modname=None)

    def get_probability(self, inputs, output):
        if tuple(inputs) not in self.evaluated:
            self.evaluate(inputs)
        i = self.output_domain.index(output)
        return float(self.evaluated[tuple(inputs)][i])


class Fact_Network(Network):

    def __init__(self, net, name, function, optimizer, model, term):
        Network.__init__(self, net, name, function, optimizer, model)
        self.term = term
        self.test_term = None

    def register_external(self, _, test_model):
        pass

    def get_probability(self, inputs):
        if tuple(inputs) not in self.evaluated:
            self.evaluate(inputs)
        return float(self.evaluated[tuple(inputs)])


class Det_Network(Network):

    def __init__(self, net, name, function, optimizer, model, term):
        Network.__init__(self, net, name, function, optimizer, model)
        if type(term) is problog.logic.Clause:
            head, body = term.head, term.body
        else:
            head, body = term, None
        self.input_terms = term2list2(head.probability.args[1])
        self.output_variable = head.probability.args[2]
        extra_term = problog.logic.Term(head.probability.args[0], *self.input_terms, self.output_variable)
        head.probability = None
        self.term = head << (extra_term & body if body else extra_term)
        self.test_term = self.term

    def predicate(self, *inputs):
        output = self.function(self,*inputs)
        return output

    def register_external(self, model, test_model):
        signature = ['+term'] * len(self.input_terms) + ['-term']
        problog_export.database = model
        problog_export(*signature)(self.predicate, funcname=self.name, modname=None)
        problog_export.database = test_model
        problog_export(*signature)(self.predicate, funcname=self.name, modname=None)
