import chainer
import chainer.functions as F
from chainer import FunctionNode, Variable
from chainer.backends import cuda
import numpy as np


class NaiveExpAdd(FunctionNode):
    """
    define function node
    z := f(x,y) = exp(x) + exp(y)
    """

    def forward_cpu(self, inputs):
        self.retain_inputs((0, 1))
        x, y = inputs
        z = np.exp(x) + np.exp(y)
        return z,

    def forward_gpu(self, inputs):
        self.retain_inputs((0, 1))
        cp = cuda.cupy
        x, y = inputs
        z = cp.exp(x) + cp.exp(y)
        return z,

    def backward(self, target_input_indexes, grad_outputs):
        x, y = self.get_retained_inputs()
        gz, = grad_outputs
        gx = F.exp(x) * gz
        gy = F.exp(y) * gz
        return gx, gy


class ExpAdd(FunctionNode):
    """
    define function node
    z := f(x,y) = exp(x) + exp(y)
    """

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xp = chainer.backend.get_array_module(*inputs)
        x, y = inputs
        z = xp.exp(x) + xp.exp(y)
        return z,

    def backward(self, target_input_indexes, grad_outputs):
        x, y = self.get_retained_inputs()
        gz, = grad_outputs
        gx = F.exp(x) * gz
        gy = F.exp(y) * gz
        return gx, gy


def naive_expadd(x, y):
    z, = NaiveExpAdd().apply((x, y))
    return z


def expadd(x, y):
    z, = ExpAdd().apply((x, y))
    return z


def main():
    x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
    y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
    naive_expadd(x, y)
    z = expadd(x, y)
    xp = chainer.backend.get_array_module(*(x, y))
    z.grad = xp.ones(z.shape, dtype=np.float32)
    z.backward()
    assert (x.grad == xp.exp(x.array)).all()
    assert (y.grad == xp.exp(y.array)).all()
    print("Done!")

if __name__ == '__main__':
    main()
