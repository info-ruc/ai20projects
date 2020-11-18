from problog.logic import Term
from network import AD_Network, Fact_Network


class VectorShape(object):

    def __init__(self, model, ground):
        networks = set()
        parameters = set()
        self.length = 0
        self.indices = list()
        self.model = model
        for k, w in ground.get_weights().items():
            if type(w) is Term and w.functor == 'nn':
                network_name = str(w.args[0])
                id = network_name, w.args[1]
                networks.add(id)
            elif type(w) is Term and w.functor == 't':
                parameters.add(w.location)
        self.networks = list(networks)
        self.parameters = list(parameters)
        self.shape = self.networks + self.parameters
        for network, _ in networks:
            network = model.networks[network]
            self.indices.append(self.length)
            if type(network) is AD_Network:
                self.length += len(network.output_domain)
            elif type(network) is Fact_Network:
                self.length += 1
            else:
                raise Exception('Unknown network type', type(network))
        for _ in parameters:
            self.indices.append(self.length)
            self.length += 1

    def split(self, vector):
        out = dict()
        i = 0
        for network, args in self.networks:
            network = self.model.networks[network]
            if type(network) is AD_Network:
                length = len(network.output_domain)
            elif type(network) is Fact_Network:
                length = 1
            out[(network.name, args)] = vector[i:i + length]
            i += length
        for loc in self.parameters:
            out[loc] = vector[i:i+1]
            i += 1
        return out

    def get_index_network_output(self, network, output):
        return self.model.networks[network].output_domain.index(output)

    def get_index_network(self, network, input):
        return self.indices[self.shape.index((network, input))]

    def get_index_parameter(self, loc):
        return self.indices[self.shape.index(loc)]