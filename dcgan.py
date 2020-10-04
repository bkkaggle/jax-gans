# from https://objax.readthedocs.io/en/latest/notebooks/Custom_Networks.html
import os
from tqdm import tqdm

import numpy as np
import tensorflow_datasets as tfds
import wandb

import jax.numpy as jnp

import objax
from objax.util import EasyDict
from objax.zoo.dnnet import DNNet
from objax import random


def normal_0_02(shape):
    return random.normal(shape, mean=0.0, stddev=0.02)


class Generator(objax.Module):
    def __init__(self):
        self.conv_block_1 = objax.nn.Sequential([
            objax.nn.ConvTranspose2D(
                100, 64*8, k=4, strides=1, padding=objax.constants.ConvPadding.VALID, use_bias=False, w_init=normal_0_02),
            objax.nn.BatchNorm2D(64*8)
        ])

        self.conv_block_2 = objax.nn.Sequential([
            objax.nn.ConvTranspose2D(
                64*8, 64*4, k=4, strides=2, padding=objax.constants.ConvPadding.SAME, use_bias=False, w_init=normal_0_02),
            objax.nn.BatchNorm2D(64*4)
        ])

        self.conv_block_3 = objax.nn.Sequential([
            objax.nn.ConvTranspose2D(
                64*4, 64*2, k=4, strides=2, padding=objax.constants.ConvPadding.SAME, use_bias=False, w_init=normal_0_02),
            objax.nn.BatchNorm2D(64*2)
        ])

        self.conv_block_4 = objax.nn.Sequential([
            objax.nn.ConvTranspose2D(
                64*2, 64, k=4, strides=2, padding=objax.constants.ConvPadding.SAME, use_bias=False, w_init=normal_0_02),
            objax.nn.BatchNorm2D(64)
        ])

        self.out_conv = objax.nn.ConvTranspose2D(
            64, 3, k=4, strides=2, padding=objax.constants.ConvPadding.SAME, w_init=normal_0_02, use_bias=False)

    def __call__(self, x, training):
        x = self.conv_block_1(x, training=training)
        x = objax.functional.relu(x)

        x = self.conv_block_2(x, training=training)
        x = objax.functional.relu(x)

        x = self.conv_block_3(x, training=training)
        x = objax.functional.relu(x)

        x = self.conv_block_4(x, training=training)
        x = objax.functional.relu(x)

        x = self.out_conv(x)
        x = objax.functional.tanh(x)

        return x


class Discriminator(objax.Module):
    def __init__(self):
        self.conv_block_1 = objax.nn.Conv2D(
            3, 64, k=4, strides=2, padding=objax.constants.ConvPadding.SAME, use_bias=False, w_init=normal_0_02)

        self.conv_block_2 = objax.nn.Sequential([
            objax.nn.Conv2D(64, 64*2, k=4, strides=2,
                            padding=objax.constants.ConvPadding.SAME, use_bias=False, w_init=normal_0_02),
            objax.nn.BatchNorm2D(64*2)
        ])

        self.conv_block_3 = objax.nn.Sequential([
            objax.nn.Conv2D(64*2, 64*4, k=4, strides=2,
                            padding=objax.constants.ConvPadding.SAME, use_bias=False, w_init=normal_0_02),
            objax.nn.BatchNorm2D(64*4)
        ])

        self.conv_block_4 = objax.nn.Sequential([
            objax.nn.Conv2D(64*4, 64*8, k=4, strides=2,
                            padding=objax.constants.ConvPadding.SAME, use_bias=False, w_init=normal_0_02),
            objax.nn.BatchNorm2D(64*8)
        ])

        self.out_conv = objax.nn.Conv2D(
            64*8, 1, k=4, strides=1, padding=objax.constants.ConvPadding.VALID, w_init=normal_0_02, use_bias=False)

    def __call__(self, x, training):
        x = self.conv_block_1(x)
        x = objax.functional.leaky_relu(x, 0.02)

        x = self.conv_block_2(x, training=training)
        x = objax.functional.leaky_relu(x, 0.02)

        x = self.conv_block_3(x, training=training)
        x = objax.functional.leaky_relu(x, 0.02)

        x = self.conv_block_4(x, training=training)
        x = objax.functional.leaky_relu(x, 0.02)

        x = self.out_conv(x)
        x = objax.functional.sigmoid(x)

        x = jnp.reshape(x, [-1, 1])

        return x


def main():
    wandb.init(project="jax-gans")

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

    generator = Generator()
    discriminator = Discriminator()

    lr = 1e-3  # learning rate
    batch = 64
    epochs = 10

    def train_model(generator, discriminator):

        g_opt = objax.optimizer.Momentum(generator.vars())
        d_opt = objax.optimizer.Momentum(discriminator.vars())

        def d_loss(x, z):
            d_loss_real = objax.functional.loss.cross_entropy_logits(
                discriminator(x, training=True), 1).mean()

            fake_img = generator(z, training=False)
            d_loss_fake = objax.functional.loss.cross_entropy_logits(
                discriminator(fake_img, training=True), 0).mean()

            d_loss = d_loss_real + d_loss_fake

            return d_loss

        def g_loss(x, z):
            fake_img = generator(z, training=True)
            return objax.functional.loss.cross_entropy_logits(discriminator(fake_img, training=False), 1).mean()

        d_gv = objax.GradValues(d_loss, discriminator.vars())
        g_gv = objax.GradValues(g_loss, generator.vars())

        def d_train_op(x, z):
            g, v = d_gv(x, z)  # returns gradients, loss
            d_opt(lr, g)
            return v

        def g_train_op(x, z):
            g, v = g_gv(x, z)  # returns gradients, loss
            g_opt(lr, g)
            return v

        d_train_op = objax.Jit(d_train_op, d_gv.vars() + d_opt.vars())
        g_train_op = objax.Jit(g_train_op, g_gv.vars() + g_opt.vars())

        for epoch in range(epochs):
            d_avg_loss = 0
            g_avg_loss = 0

            shuffle_idx = np.random.permutation(train.image.shape[0])
            for i, it in tqdm(enumerate(range(0, train.image.shape[0], batch))):
                sel = shuffle_idx[it: it + batch]

                z = random.normal([batch, 100, 1, 1])
                img = train.image[sel]

                d_loss = float(d_train_op(
                    train.image[sel], z)[0]) * len(sel)
                g_loss = float(g_train_op(
                    train.image[sel], z)[0]) * len(sel)

                g_avg_loss += g_loss
                d_avg_loss += d_loss

                if i % 10 == 0:
                    wandb.log({"g_loss": g_loss,
                               "d_loss": d_loss}, step=(epoch + 1) * (i + 1))

            d_avg_loss /= it + len(sel)
            g_avg_loss /= it + len(sel)

            print('Epoch %04d d Loss %.2f g Loss %.2f' %
                  (epoch + 1, d_avg_loss, g_avg_loss))

    train_model(generator, discriminator)


if __name__ == "__main__":
    main()
