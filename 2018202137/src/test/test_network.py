import unittest
import problog
from network import Network
from standard_networks import FC

parser = problog.parser.PrologParser(problog.program.ExtendedPrologFactory())


class TestNetwork(unittest.TestCase):

    def test_ad_instantiate(self):
        fc = FC(10,2)
        net = Network(fc, 'fc', None)
        in_line = 'nn(fc,[X,Y],Z,[0,1])::term(X,Y,Z) :- a(Y).'
        out_line = 'nn(fc,[X,Y],0)::term(X,Y,0);nn(fc,[X,Y],1)::term(X,Y,1) :- a(Y).'
        test_out_line = 'nn(fc,[X,Y],Z)::term(X,Y,Z) :- fc(X,Y,Z), a(Y).'
        term = parser.parseString(in_line)[0]
        expected = parser.parseString(out_line)[0]
        test_expected = parser.parseString(test_out_line)[0]
        net = net.instantiate(term)
        self.assertEqual(str(net.term), str(expected))
        self.assertEqual(str(net.test_term), str(test_expected))

        in_line = 'nn(fc,[X,Y],Z,[0,1])::term(X,Y,Z).'
        out_line = 'nn(fc,[X,Y],0)::term(X,Y,0);nn(fc,[X,Y],1)::term(X,Y,1).'
        test_out_line = 'nn(fc,[X,Y],Z)::term(X,Y,Z) :- fc(X,Y,Z).'
        term = parser.parseString(in_line)[0]
        expected = parser.parseString(out_line)[0]
        test_expected = parser.parseString(test_out_line)[0]
        net = net.instantiate(term)
        self.assertEqual(str(net.term), str(expected))
        self.assertEqual(str(net.test_term), str(test_expected))

    def test_det_instantiate(self):
        fc = FC(10,2)
        net = Network(fc, 'fc', None)
        in_line = 'nn(fc,[X,Y],Z)::term(X,Y,Z) :- a(Y).'
        out_line = 'term(X,Y,Z) :- fc(X,Y,Z), a(Y).'
        term = parser.parseString(in_line)[0]
        expected = parser.parseString(out_line)[0]
        net = net.instantiate(term)
        self.assertEqual(str(net.term), str(expected))

        in_line = 'nn(fc,[X,Y],Z)::term(X,Y,Z).'
        out_line = 'term(X,Y,Z) :- fc(X,Y,Z).'
        term = parser.parseString(in_line)[0]
        expected = parser.parseString(out_line)[0]
        net = net.instantiate(term)
        self.assertEqual(str(net.term), str(expected))

    def test_fact_instantiate(self):
        fc = FC(10,2)
        net = Network(fc, 'fc', None)
        in_line = 'nn(fc,[X,Y])::term(X,Y) :- a(Y).'
        out_line = 'nn(fc,[X,Y])::term(X,Y) :-  a(Y).'
        term = parser.parseString(in_line)[0]
        expected = parser.parseString(out_line)[0]
        net = net.instantiate(term)
        self.assertEqual(str(net.term), str(expected))

        in_line = 'nn(fc,[X,Y])::term(X,Y).'
        out_line = 'nn(fc,[X,Y])::term(X,Y).'
        term = parser.parseString(in_line)[0]
        expected = parser.parseString(out_line)[0]
        net = net.instantiate(term)
        self.assertEqual(str(net.term), str(expected))


if __name__== '__main__':
    unittest.main()