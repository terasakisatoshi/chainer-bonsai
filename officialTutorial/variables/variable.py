import numpy as np
import chainer

x_data = np.array([5], dtype=np.float32)
x = chainer.Variable(x_data)
print(x.grad)
y = x**2 - 2 * x + 1
print(y.array)
print(y.grad)
y.backward()
print(x.grad)
z = 2 * x
y = x**2 - z + 1
y.backward()
print(z.grad)  # None
"""
Note that Chainer, by default, releases the gradient arrays of intermediate variables for memory efficiency.
In order to preserve gradient information, pass the retain_grad argument to the backward method:
"""
y.backward(retain_grad=True)
print(z.grad)

x = chainer.Variable(
    np.array(
        [[1, 2, 3],
         [4, 5, 6]],
        dtype=np.float32
    )
)

y = x**2 - 2 * x + 1
z = y[:, 0:2]
z.grad = np.ones(z.shape, dtype=np.float32)
z.backward()
print(x.grad)
