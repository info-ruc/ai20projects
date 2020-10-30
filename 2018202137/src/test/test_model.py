from unittest import TestCase
from model import Model
from standard_networks import FC
from network import Network
from problog.logic import Term, Constant, Var
import torch
from torch.autograd import Variable

def test(net, input):
    return Variable(torch.FloatTensor([float(input)]))


class TestModel(TestCase):

    def __init__(self, *args):
        TestCase.__init__(self, *args)


    def test_save_load(self):
        model_string = "nn(fc,[X,Y],Z,[0,1]) :: a(X,Y,Z).\nt(0.5)::b."
        fc = FC(10, 2)
        net = Network(fc, 'fc', lambda a, b, c: Variable(torch.FloatTensor([0.2, 0.8])))
        model = Model(model_string, [net])
        model.save_state('test_save.mdl')
        orig_params = dict()
        for p in model.parameters:
            orig_params[p] = model.parameters[p]
            model.parameters[p] = 0
        original_net_params = list()

        for i, param in enumerate(fc.parameters()):
            original_net_params.append(param.data.clone())
            param.data.zero_()
        model.load_state('test_save.mdl')
        for p in model.parameters:
            self.assertEqual(model.parameters[p],orig_params[p])

        for i, param in enumerate(fc.parameters()):
            self.assertTrue(torch.equal(param.data, original_net_params[i]))

    def test_sdd_disjoin(self):
        model_string = "0.8::a.\nb:-a.\nb:-c.\n0.1::c."
        model = Model(model_string,[])
        query = Term('b')
        solution = model.solve(query)
        self.assertAlmostEqual(solution[query][0], 0.82)

    def test_strings(self):
        model_string = "nn(fc,[X,Y],Z,[0,1]) :: a(X,Y,Z).\nb(X,Y,Z) :- a(X,Y,Z)."
        fc = FC(10, 2)
        net = Network(fc,'fc', lambda a, b, c: Variable(torch.FloatTensor([0.2,0.8])))
        model = Model(model_string, [net])
        query = Term('b',Constant("string with 's"), Constant(3), Var('X'))
        solution = model.solve(query)
        print(solution)

    def test_constraint(self):
        model_string = "nn(fc,[X])::a(X).\nb:-a(0.2);a(0.8).\naux:-a(0.2),a(0.8).\naux2:-a(0.2);a(0.8).\nevidence(aux2).\nevidence(aux,false)."
        fc = FC(10, 2)
        net = Network(fc, 'fc', test)
        model = Model(model_string, [net])
        query = Term('b')
        solution = model.solve(query)
        print(solution)
        #self.assertAlmostEqual(solution[query][0],1)

    def test_indirect_evidence(self):
        model_string="""nn(fc,[X],Y,[0,1,2,3,4,5,6,7,8,9])::digit(X,Y).
addition(X,Y,Z) :- digit(X,X2),digit(Y,Y2),Z is X2+Y2."""
#evidence(digit(2,2))."""
        #model_string = "nn(fc,[X])::a(X).\nb:-a(0.2).\nc:-b.\nevidence(b)."
        fc = FC(10, 10)
        net = Network(fc, 'fc', lambda a,b: Variable(torch.FloatTensor([0.1]*10)))
        model = Model(model_string, [net])
        query = Term('digit',Constant(2),Constant(2))
        #query = Term('addition',Constant(2),Constant(3),Constant(5))
        solution = model.solve(query)
        print(solution)

    def test_test(self):
        model_string = "nn(fc,[X,Y],Z,[0,1]) :: a(X,Y,Z).\nb(X,Y,Z) :- a(X,Y,Z)."
        fc = FC(10, 2)
        net = Network(fc,'fc', lambda a, b, c: Variable(torch.FloatTensor([0.2,0.8])))
        model = Model(model_string, [net])
        query = Term('b',Constant("string with 's"), Constant(3), Var('X'))
        solution = model.solve(query,test=True)
        print(solution)