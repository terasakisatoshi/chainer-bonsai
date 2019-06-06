import chainer
import chainer.functions as F
import chainer.links as L
from chainer import iterators
from chainer.datasets import mnist
from chainer import training
from chainer.training import extensions


class MLP(chainer.Chain):
    def __init__(self, n_hidden=100, n_out=10):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_hidden)
            self.l2 = L.Linear(None, n_hidden)
            self.l3 = L.Linear(None, n_out)

    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        return h


def train():
    train, test = mnist.get_mnist()
    batchsize = 128
    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(
        test, batchsize, shuffle=False, repeat=False)
    model = L.Classifier(MLP())

    device = -1
    max_epoch = 10
    if chainer.backends.cuda.available:
        device = 0
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out="mnist_result")
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.snapshot(filename="snapshot_epoch-{.updater.epoch}"))
    trainer.extend(extensions.snapshot_object(model, filename="model_epoch-{.updater.epoch}"))
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))
    trainer.extend(extensions.PrintReport(["epoch", "main/loss", "main/accuracy",
                                           "validation/main/loss", "validation/main/accuracy", "elapsed_time"]))
    trainer.extend(extensions.PlotReport(["main/loss", "validation/main/loss"], x_key="epoch", file_name="loss.png"))
    trainer.extend(extensions.PlotReport(["main/accuracy", "validation/main/accuracy"], x_key="epoch", file_name="accuracy"))
    trainer.extend(extensions.DumpGraph("main/loss"))
    trainer.run()


def test():
    import matplotlib.pyplot as plt
    import numpy as np
    model = L.Classifier(MLP())
    chainer.serializers.load_npz("mnist_result/model_epoch-10", model)
    train, test = mnist.get_mnist()
    x, t = test[0]
    y = model.predictor(x[np.newaxis, ...])
    predicted_number = np.squeeze(y.array).argmax()
    print(predicted_number)
    print("actual {}".format(t))
    plt.imshow(x.reshape(28, 28), cmap="gray")
    plt.savefig("mnist_result/test0.png")


def main():
    train()
    test()


if __name__ == '__main__':
    main()
