import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import mnist
from chainer.dataset import concat_examples
from chainer.iterators import SerialIterator
from chainer import optimizers
import numpy as np
"""
In this code, we will learn how to train a deep neural network
to classify images of hand-written digits in the popular MNIST
dataset. This dataset contains 50,000 training example and 10,000
test examples.
Each Example is a set of 28x28 gray scale image and a corresponding
class label. Since the digits from 0 to 9 are used, there are 10
classes for the labels.
"""

"""
Chainer provides Trainer class that can simplify the training 
proceduce of our model. Howeve, it is also useful how the training
loop works in Chainer.
Writing your own training loop can be useful for learning how Trainer
works or for implementing feature not included in the standard trainer.
"""


class MyNetwork(chainer.Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(MyNetwork, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h


def train():
    batchsize = 128
    max_epoch = 10
    device = 0
    train_data, test_data = mnist.get_mnist(withlabel=True, ndim=1)
    train_iter = SerialIterator(train_data, batchsize)
    test_iter = SerialIterator(test_data, batchsize, repeat=False, shuffle=False)
    model = MyNetwork()
    if chainer.cuda.available and device >= 0:
        model.to_gpu(device)
    else:
        device = -1
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    while train_iter.epoch < max_epoch:
        train_batch = train_iter.next()
        image_train, target_train = concat_examples(train_batch, device)
        prediction_train = model(image_train)
        loss = F.softmax_cross_entropy(prediction_train, target_train)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        if train_iter.is_new_epoch:
            loss_array = float(chainer.backends.cuda.to_cpu(loss.array))
            print("epoch{:2d} train_loss:{:.04f}".format(train_iter.epoch, loss_array))
            test_losses = []
            test_accs = []
            while True:
                test_batch = test_iter.next()
                image_test, target_test = concat_examples(test_batch, device)
                prediction_test = model(image_test)
                loss_test = F.softmax_cross_entropy(prediction_test, target_test)
                test_losses.append(chainer.backends.cuda.to_cpu(loss_test.array))
                acc = F.accuracy(prediction_test, target_test)
                test_accs.append(chainer.backends.cuda.to_cpu(acc.array))
                if test_iter.is_new_epoch:
                    test_iter.reset()
                    break
            mean_loss = np.mean(test_losses)
            mean_acc = np.mean(test_accs)
            print("val_loss:{:.04f} val_accuracy:{:.04f}".format(mean_loss, mean_acc))

    chainer.serializers.save_npz("model.npz", model)


def test():
    model = MyNetwork()
    chainer.serializers.load_npz("model.npz", model)
    _, test_data = mnist.get_mnist(withlabel=True, ndim=1)
    x, t = test_data[0]
    y = model(np.expand_dims(x, axis=0))
    y = np.squeeze(y.array)
    pred_label = y.argmax()
    print("prediction", pred_label)
    print("actual", t)
    if pred_label == t:
        print("Yay")


if __name__ == '__main__':
    train()
    test()
