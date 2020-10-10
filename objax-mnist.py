# from https://objax.readthedocs.io/en/latest/notebooks/Custom_Networks.html
import os
from tqdm import tqdm

import numpy as np
import tensorflow_datasets as tfds

import objax
from objax.util import EasyDict
from objax.zoo.dnnet import DNNet


class ConvNet(objax.Module):
    """ConvNet implementation."""

    def __init__(self, nin, nclass):
        """Define 3 blocks of conv-bn-relu-conv-bn-relu followed by linear layer."""
        self.conv_block1 = objax.nn.Sequential([objax.nn.Conv2D(nin, 16, 3, use_bias=False),
                                                objax.nn.BatchNorm2D(16),
                                                objax.functional.relu,
                                                objax.nn.Conv2D(
                                                    16, 16, 3, use_bias=False),
                                                objax.nn.BatchNorm2D(16),
                                                objax.functional.relu])
        self.conv_block2 = objax.nn.Sequential([objax.nn.Conv2D(16, 32, 3, use_bias=False),
                                                objax.nn.BatchNorm2D(32),
                                                objax.functional.relu,
                                                objax.nn.Conv2D(
                                                    32, 32, 3, use_bias=False),
                                                objax.nn.BatchNorm2D(32),
                                                objax.functional.relu])
        self.conv_block3 = objax.nn.Sequential([objax.nn.Conv2D(32, 64, 3, use_bias=False),
                                                objax.nn.BatchNorm2D(64),
                                                objax.functional.relu,
                                                objax.nn.Conv2D(
                                                    64, 64, 3, use_bias=False),
                                                objax.nn.BatchNorm2D(64),
                                                objax.functional.relu])
        self.linear = objax.nn.Linear(64, nclass)

    def __call__(self, x, training):
        x = self.conv_block1(x, training=training)
        x = objax.functional.max_pool_2d(x, size=2, strides=2)
        x = self.conv_block2(x, training=training)
        x = objax.functional.max_pool_2d(x, size=2, strides=2)
        x = self.conv_block3(x, training=training)
        x = x.mean((2, 3))
        x = self.linear(x)
        return x


def main():
        # Data: train has 60000 images - test has 10000 images
        # Each image is resized and converted to 32 x 32 x 3
    DATA_DIR = os.path.join(os.environ['HOME'], 'TFDS')
    data = tfds.as_numpy(
        tfds.load(name='mnist', batch_size=-1, data_dir=DATA_DIR))

    def prepare(x):
        """Pads 2 pixels to the left, right, top, and bottom of each image, scales pixel value to [-1, 1], and converts to NCHW format."""
        s = x.shape
        x_pad = np.zeros((s[0], 32, 32, 1))
        x_pad[:, 2:-2, 2:-2, :] = x
        return objax.util.image.nchw(
            np.concatenate([x_pad.astype('f') * (1 / 127.5) - 1] * 3, axis=-1))

    train = EasyDict(image=prepare(
        data['train']['image']), label=data['train']['label'])
    test = EasyDict(image=prepare(
        data['test']['image']), label=data['test']['label'])
    ndim = train.image.shape[-1]

    del data

    cnn_model = ConvNet(nin=3, nclass=10)
    print(cnn_model.vars())

    # Settings
    lr = 0.03  # learning rate
    batch = 128
    epochs = 100

    # Train loop

    def train_model(model):

        def predict(model, x):
            """"""
            return objax.functional.softmax(model(x,  training=False))

        def flatten_image(x):
            """Flatten the image before passing it to the DNN."""
            if isinstance(model, DNNet):
                return objax.functional.flatten(x)
            else:
                return x

        opt = objax.optimizer.Momentum(model.vars())

        # Cross Entropy Loss
        def loss(x, label):
            return objax.functional.loss.cross_entropy_logits_sparse(model(x, training=True), label).mean()

        gv = objax.GradValues(loss, model.vars())

        def train_op(x, label):
            g, v = gv(x, label)  # returns gradients, loss
            opt(lr, g)
            return v

        train_op = objax.Jit(train_op, gv.vars() + opt.vars())

        for epoch in range(epochs):
            avg_loss = 0
            # randomly shuffle training data
            shuffle_idx = np.random.permutation(train.image.shape[0])
            for it in tqdm(range(0, train.image.shape[0], batch)):
                sel = shuffle_idx[it: it + batch]
                avg_loss += float(train_op(flatten_image(
                    train.image[sel]), train.label[sel])[0]) * len(sel)
            avg_loss /= it + len(sel)

            # Eval
            accuracy = 0
            for it in tqdm(range(0, test.image.shape[0], batch)):
                x, y = test.image[it: it + batch], test.label[it: it + batch]
                accuracy += (np.argmax(predict(model, flatten_image(x)),
                                       axis=1) == y).sum()
            accuracy /= test.image.shape[0]
            print('Epoch %04d  Loss %.2f  Accuracy %.2f' %
                  (epoch + 1, avg_loss, 100 * accuracy))

    train_model(cnn_model)


if __name__ == "__main__":
    main()
