import problog
import os
from problog.logic import *
from problog.program import PrologString
from logic.gradient_semiring import solve, extract_parameters
from logic.vector_shape import VectorShape
from problog.sdd_formula import SDD
from zipfile import ZipFile
import pickle
from logic.gradient_semiring import SemiringGradient

class Model(object):

    def __init__(self, model_string, networks, caching=False, saving=False):
        self.networks = dict()
        for network in networks: # 这里写的东西就离谱
            self.networks[network.name] = network
            network.model = self
        self.model_string = self.parse(model_string)
        self.engine = problog.engine.DefaultEngine(), problog.engine.DefaultEngine()
        train_model = self.engine[0].prepare(PrologString(self.model_string[0] + '\n' + self.model_string[1]))
        test_model = self.engine[1].prepare(PrologString(self.model_string[0] + '\n' + self.model_string[2]))
        self.problog_model = train_model, test_model

        for network in self.networks.values():
            network.register_external(*self.problog_model)
        self.sdd_manager = None
        self.parameters, self.ADs = extract_parameters(train_model)
        self.caching = caching
        self.saving = saving
        self.n = 0
        self.obj_store = list()
        if caching:
            self.sdd_cache = dict()
        if saving:
            import os
            if not os.path.exists('sdd/'):
                os.makedirs('sdd/')

    def parse(self, model_string):
        new_lines_train, new_lines_test, original_model = list(), list(), list()
        parser = problog.parser.PrologParser(problog.program.ExtendedPrologFactory())
        for line in model_string.split('\n'):
            if '::' in line and line[:2] == 'nn':
                parsed = parser.parseString(line.rstrip())
                for term in parsed:
                    annotation = term.probability.args
                    network = self.networks[str(annotation[0])]
                    network = network.instantiate(term)
                    new_lines_train.append(str(network.term)+'.')
                    if network.test_term:
                        new_lines_test.append(str(network.test_term)+'.')
                    self.networks[str(annotation[0])] = network
            else:
                original_model.append(line)
        return '\n'.join(original_model), '\n'.join(new_lines_train), '\n'.join(new_lines_test)

    def build_sdd(self, q, test):
        i = 1 if test else 0
        ground = self.engine[i].ground_all(self.problog_model[i], queries=[q])
        sdd = SDD.create_from(ground)
        # print(sdd)
        # sdd.build_dd()
        # print(sdd)
        # shape = VectorShape(self, sdd)
        # semiring = SemiringGradient(self, shape)
        # evaluator = sdd.get_evaluator(semiring=semiring)
        return sdd

    def get_sdd(self, q, test=False):
        if self.caching and not test:
            if str(q) not in self.sdd_cache:
                if self.saving:
                    fname = os.path.abspath('sdd/' + str(q))
                    try:
                        with open(fname, 'rb') as f:
                            sdd = pickle.load(f)
                    except IOError:
                        sdd = self.build_sdd(q,test)
                        with open(fname, 'wb') as f:
                            pickle.dump(sdd, f)
                else:
                    sdd = self.build_sdd(q, test)
                shape = VectorShape(self, sdd)
                self.sdd_cache[str(q)] = sdd, shape
            return self.sdd_cache[str(q)]
        else:
            # if evidence is not None:
            #     ground = self.engine[i].ground_all(self.problog_model[i], queries=[q], evidence=evidence)
            # else:
            #     ground = self.engine[i].ground_all(self.problog_model[i], queries=[q])
            # sdd = SDD.create_from(ground)
            sdd = self.build_sdd(q, test)
            shape = VectorShape(self, sdd)
            return sdd, shape

    def solve(self, query, evidence=None, test=False): # ?
        self.n += 1
        sdd, shape = self.get_sdd(query, test)
        solution = solve(self, sdd, shape)
        self.clear()
        return solution

    def accuracy(self, data, nr_output=1, test=False, verbose=False):
        correct = 0
        for d in data:
            args = list(d.args)
            args[-nr_output:] = [Var('X_{}'.format(i)) for i in range(nr_output)]
            q = d(*args)
            out = self.solve(q, None, test)
            out = max(out, key=lambda x: out[x][0])
            if out == d:
                correct += 1
            else:
                if verbose:
                    print('Wrong', d, 'vs', out)
        print('Accuracy', correct / len(data))
        return [('Accuracy', correct / len(data))]

    def save_state(self, location):
        with ZipFile(location,'w') as zipf:
            with zipf.open('parameters','w') as f:
                pickle.dump(self.parameters,f)
            for n in self.networks:
                with zipf.open(n,'w') as f:
                    self.networks[n].save(f)

    def load_state(self, location):
        with ZipFile(location) as zipf:
            with zipf.open('parameters') as f:
                self.parameters = pickle.load(f)
            for n in self.networks:
                with zipf.open(n) as f:
                    self.networks[n].load(f)

    def store(self, object):
        self.obj_store.append(object)
        return len(self.obj_store) - 1

    def retrieve(self, id):
        return self.obj_store[id]

    def clear(self):
        self.obj_store = []
