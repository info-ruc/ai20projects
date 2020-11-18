import problog
from problog.logic import *
from gradient_semiring import SemiringGradient


def extract_parameters(model):
    parameters = dict()
    ad = dict()
    for n in model.iter_nodes():
        node_type = type(n).__name__
        if node_type == 'clause' or node_type == 'fact':
            annotation = n.probability
            if annotation is not None and type(annotation) is Term and annotation.functor is 't':
                parameters[annotation.location] = float(annotation.args[0])
        elif node_type == 'choice':
            if type(n.probability) is Term and n.probability.functor == 't':
                if n.group not in ad:
                    ad[n.group] = list()
                ad[n.group].append(n.probability.location)
    return parameters, ad


def solve(model, sdd, shape):
    semiring = SemiringGradient(model, shape)
    result = sdd.evaluate(semiring=semiring)
    result = {k: (result[k][0], shape.split(result[k][1])) for k in result}
    return result


def term2list2(term):
    result = []
    while not problog.logic.is_variable(term) and term.functor == '.' and term.arity == 2:
        result.append(term.args[0])
        term = term.args[1]
    if not term == problog.logic.Term('[]'):
        raise ValueError('Expected fixed list.')
    return result