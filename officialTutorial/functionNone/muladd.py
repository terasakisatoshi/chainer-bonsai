import chainer
from chainer import FunctionNode, Variable
import numpy as np


class MulAdd(FunctionNode):
    """
    f(x, y, z) = x * y + z
    """

    def forward_cpu(self, inputs):
        # do forward computation on CPU
        #  Note that all arrays appearing in forward_cpu method are `numpy.ndarray`
        x, y, z = inputs
        self.retain_inputs((0, 1))
        w = x * y + z
        # Be careful to return a tuple even if you have just one array or Variable to return.
        return w,

    def forward_gpu(self, inputs):
        """
        In forward_gpu method, arrays are of type cupy.ndarray.
        We use arithmetic operators defined for this class.
        These operators implement the basic elementwise arithmetics.
        """
        x, y, z = inputs
        self.retain_inputs((0, 1))
        w = x * y + z
        return w,

    def backward(self, target_input_indexes, grad_outputs):
        x, y = self.get_retained_inputs()
        gw, = grad_outputs
        gx = y * gw
        gy = x * gw
        gz = gw
        return gx, gy, gz


def muladd(x, y, z):
    w, = MulAdd().apply((x, y, z))
    return w


def main():
    x = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
    y = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
    z = Variable(np.random.uniform(-1, 1, (3, 2)).astype(np.float32))
    inputs = (x, y, z)
    w = muladd(*inputs)
    xp = chainer.backend.get_array_module(*inputs)
    w.grad = xp.ones(w.shape, dtype=np.float32)
    w.backward()
    assert (x.grad == y.array).all()
    assert (y.grad == x.array).all()
    assert (z.grad == xp.ones(z.shape)).all()
    print("finished")

if __name__ == '__main__':
    main()
