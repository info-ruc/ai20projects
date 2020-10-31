class Optimizer(object):

    def __init__(self, model, accumulation):
        self.model = model
        self.accumulation = accumulation
        self.params_grad = dict()
        self.i = 0
        self.epoch = 0
        self.clear()

    def step_epoch(self):
        self.epoch += 1

    def clear(self):
        for n in self.model.networks:
            self.model.networks[n].clear()

    def add_param_grad(self, k, grad):
        if k not in self.params_grad:
            self.params_grad[k] = 0
        self.params_grad[k] += grad

    def backward(self, name, i, grad):
        self.model.networks[name].backward(i, grad)

    def step(self):
        self.i += 1
        if self.i % self.accumulation == 0:
            for _, network in self.model.networks.items():
                network.step()
            self.clear()


class SGD(Optimizer):

    def __init__(self, model, accumulation=1, param_lr=0):
        Optimizer.__init__(self, model, accumulation)
        self.param_lr = param_lr

    def get_lr(self):
        return self.param_lr

    def step(self):
        Optimizer.step(self)

        if self.i % self.accumulation == 0:
            for k in self.params_grad.keys():
                self.model.parameters[k] -= self.get_lr() * self.params_grad[k] / self.accumulation
                self.model.parameters[k] = max(min(self.model.parameters[k], 1.0), 0.0)
                self.params_grad[k] = 0

        for ad in self.model.ADs:
            ad_sum = 0
            for k in self.model.ADs[ad]:
                ad_sum += self.model.parameters[k]
            for k in self.model.ADs[ad]:
                self.model.parameters[k] /= ad_sum


