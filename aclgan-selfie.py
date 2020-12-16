import os
import argparse
from glob import glob
from functools import partial
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import cv2
from PIL import Image

import wandb

import jax
import jax.numpy as jnp

import flax
from flax import linen as nn

import torch
import torchvision.transforms as transforms
import tensorflow as tf
import tensorflow_datasets as tfds


class GANDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args

        self.paths_a = glob('./data/trainA/*.jpg')
        self.paths_b = glob('./data/trainB/*.jpg')

        self.imgs_a = np.zeros(
            (len(self.paths_a), 32, 32, 3), dtype=np.float32)
        for i, path in tqdm(enumerate(self.paths_a)):
            img = np.asarray(Image.open(path)) / 255.0
            img = cv2.resize(img, dsize=(
                32, 32), interpolation=cv2.INTER_CUBIC).reshape(1, 32, 32, 3)
            self.imgs_a[i] = img

        self.imgs_b = np.zeros(
            (len(self.paths_b), 32, 32, 3), dtype=np.float32)
        for i, path in tqdm(enumerate(self.paths_b)):
            img = np.asarray(Image.open(path)) / 255.0
            img = cv2.resize(img, dsize=(
                32, 32), interpolation=cv2.INTER_CUBIC).reshape(1, 32, 32, 3)
            self.imgs_b[i] = img

        self.transforms = transforms.Compose([
            # transforms.ToTensor(),
            # transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(hue=0.15),
            # transforms.RandomGrayscale(p=0.25),
            # transforms.RandomRotation(35),
            # transforms.RandomPerspective(distortion_scale=0.35),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.paths_a)

    def __getitem__(self, index):
        img_a = self.imgs_a[index]
        img_b = self.imgs_b[index]

        img_a = self.transforms(img_a).numpy().transpose(1, 2, 0)
        img_b = self.transforms(img_b).numpy().transpose(1, 2, 0)
        # img_a = img_a.transpose(1, 2, 0)
        # img_b = img_b.transpose(1, 2, 0)

        return img_a, img_b


def shard(xs):
    return jax.tree_map(
        lambda x: x.reshape((jax.device_count(), -1) + x.shape[1:]), xs)


class IN(nn.Module):
    def __call__(self, x):
        # assuming an image in the format HxWxC
        mu = jnp.mean(x, axis=(0, 1), keepdims=True)
        sigma_2 = jnp.var(x, axis=(0, 1), keepdims=True)

        return (x - mu) / jnp.sqrt(sigma_2 + 1e-5)


class AdaIN(nn.Module):
    @nn.compact
    # assuming an image in the format HxWxC
    def __call__(self, content, style_mean, style_var):
        content_mu = jnp.mean(content, axis=(0, 1), keepdims=True)
        content_sigma_2 = jnp.var(content, axis=(0, 1), keepdims=True)

        normalized_content = (content - content_mu) / \
            jnp.sqrt(content_sigma_2 + 1e-5)
        return normalized_content * jnp.sqrt(style_var + 1e-5) + style_mean


class Block(nn.Module):
    padding: int
    features: int
    kernel_size: int
    strides: int
    norm: str = "none"

    @nn.compact
    def __call__(self, x, means=None, variances=None):
        x = jnp.pad(x, [(self.padding, self.padding),
                        (self.padding, self.padding), (0, 0)], mode='constant')

        if self.norm == "IN":
            x = IN()(x)
        elif self.norm == 'AdaIN' and means is not None and variances is not None:
            x = AdaIN()(x, means, variances)

        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(self.kernel_size, self.kernel_size),
                    strides=(self.strides, self.strides), padding='VALID')(x)

        return x


class ResBlock(nn.Module):
    padding: int
    features: int
    kernel_size: int
    strides: int
    norm: str = "none"

    @nn.compact
    def __call__(self, x, means=None, variances=None):
        residual_branch = x

        x = Block(self.padding, self.features,
                  self.kernel_size, self.strides, self.norm)(x, means, variances)
        x = Block(self.padding, self.features,
                  self.kernel_size, self.strides, "none")(x, means, variances)

        return x + residual_branch


class StyleEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = Block(padding=3, features=64, kernel_size=7, strides=1)(x)

        x = Block(padding=1, features=64*2, kernel_size=4, strides=2)(x)
        x = Block(padding=1, features=64*2*2,
                  kernel_size=4, strides=2)(x)

        x = Block(padding=1, features=64*2*2,
                  kernel_size=4, strides=2)(x)
        x = Block(padding=1, features=64*2*2,
                  kernel_size=4, strides=2)(x)

        x = jnp.mean(x, axis=(0, 1), keepdims=True)
        x = nn.Conv(features=8, kernel_size=(1, 1),
                    strides=(1, 1), padding='VALID')(x)

        return x


class ContentEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):

        x = Block(padding=3, features=64, kernel_size=7,
                  strides=1, norm='IN')(x)

        x = Block(padding=1, features=64*2,
                  kernel_size=4, strides=2, norm='IN')(x)
        x = Block(padding=1, features=64*2*2,
                  kernel_size=4, strides=2, norm='IN')(x)

        x = ResBlock(
            padding=1, features=64*2*2, kernel_size=3, strides=1, norm='IN')(x)
        x = ResBlock(
            padding=1, features=64*2*2, kernel_size=3, strides=1, norm='IN')(x)
        x = ResBlock(
            padding=1, features=64*2*2, kernel_size=3, strides=1, norm='IN')(x)
        x = ResBlock(
            padding=1, features=64*2*2, kernel_size=3, strides=1, norm='IN')(x)

        return x


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=256, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=4*256*2, use_bias=True)(x)

        return x


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x, means, variances):
        x = ResBlock(padding=1, features=64*2*2, kernel_size=3,
                     strides=1, norm='AdaIN')(x, means[0], variances[0])
        x = ResBlock(padding=1, features=64*2*2, kernel_size=3,
                     strides=1, norm='AdaIN')(x, means[1], variances[1])
        x = ResBlock(padding=1, features=64*2*2, kernel_size=3,
                     strides=1, norm='AdaIn')(x, means[2], variances[2])
        x = ResBlock(padding=1, features=64*2*2, kernel_size=3,
                     strides=1, norm='AdaIn')(x, means[3], variances[3])

        x = jax.image.resize(
            x, (x.shape[0]*2, x.shape[1]*2, x.shape[2]), method=jax.image.ResizeMethod.NEAREST)
        x = Block(padding=2, features=64*2, kernel_size=5,
                  strides=1, norm='IN')(x)

        x = jax.image.resize(
            x, (x.shape[0]*2, x.shape[1]*2, x.shape[2]), method=jax.image.ResizeMethod.NEAREST)
        x = Block(padding=2, features=64, kernel_size=5,
                  strides=1, norm='IN')(x)

        x = Block(padding=3, features=3, kernel_size=7, strides=1)(x)
        x = jnp.tanh(x)

        return x


class Generator(nn.Module):
    @nn.compact
    def __call__(self, x):
        style = StyleEncoder()(x)
        content = ContentEncoder()(x)

        AdaIN_params = MLP()(style)

        means = []
        variances = []
        for i in range(0, 4*256*2, 256*2):
            means_variances = AdaIN_params[:, :, i:i+256*2]
            means.append(means_variances[:, :, :256])
            variances.append(means_variances[:, :, 256:])

        out = Decoder()(content, means, variances)

        return out


class Discriminator(nn.Module):
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(
            4, 4), strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(features=64*2, kernel_size=(4, 4),
                    strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(
            use_running_average=not self.training, momentum=0.9)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(features=64*4, kernel_size=(4, 4),
                    strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(
            use_running_average=not self.training, momentum=0.9)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(features=64*8, kernel_size=(4, 4),
                    strides=(2, 2), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(
            use_running_average=not self.training, momentum=0.9)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)

        x = nn.Conv(features=1, kernel_size=(
            1, 1), strides=(4, 4), padding='VALID', use_bias=False)(x)
        x = jnp.reshape(x, [x.shape[0], -1])

        return x


@jax.vmap
def bce_logits_loss(logit, label):
    return jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(-jnp.abs(logit)))


def loss_g(params_g, params_d, batch, rng, variables_g, variables_d):
    z = jax.random.normal(rng, shape=(batch.shape[0], 1, 1, 100))

    fake_batch, variables_g = Generator(training=True).apply(
        {'params': params_g, 'batch_stats': variables_g['batch_stats']}, z, mutable=['batch_stats'])

    fake_logits, variables_d = Discriminator(training=True).apply(
        {'params': params_d, 'batch_stats': variables_d['batch_stats']}, fake_batch, mutable=['batch_stats'])

    real_labels = jnp.ones((batch.shape[0],), dtype=jnp.int32)
    return jnp.mean(bce_logits_loss(fake_logits, real_labels)), (variables_g, variables_d)


def loss_d(params_d, params_g, batch, rng, variables_g, variables_d):
    z = jax.random.normal(rng, shape=(batch.shape[0], 1, 1, 100))

    fake_batch, variables_g = Generator(training=True).apply(
        {'params': params_g, 'batch_stats': variables_g['batch_stats']}, z, mutable=['batch_stats'])

    real_logits, variables_d = Discriminator(training=True).apply(
        {'params': params_d, 'batch_stats': variables_d['batch_stats']}, batch, mutable=['batch_stats'])
    fake_logits, variables_d = Discriminator(training=True).apply(
        {'params': params_d, 'batch_stats': variables_d['batch_stats']}, fake_batch, mutable=['batch_stats'])

    real_labels = jnp.ones((batch.shape[0],), dtype=jnp.int32)
    real_loss = bce_logits_loss(real_logits, real_labels)

    fake_labels = jnp.zeros((batch.shape[0],), dtype=jnp.int32)
    fake_loss = bce_logits_loss(fake_logits, fake_labels)

    return jnp.mean(real_loss + fake_loss), (variables_g, variables_d)


@partial(jax.pmap, axis_name='batch')
def train_step(rng, variables_g, variables_d, optimizer_g, optimizer_d, batch):
    rng, rng_g, rng_d = jax.random.split(rng, 3)

    (g_loss, (variables_g, variables_d)), grad_g = jax.value_and_grad(loss_g, has_aux=True)(
        optimizer_g.target, optimizer_d.target, batch, rng_g, variables_g, variables_d)
    g_loss = jax.lax.pmean(g_loss, axis_name='batch')
    grad_g = jax.lax.pmean(grad_g, axis_name='batch')

    optimizer_g = optimizer_g.apply_gradient(grad_g)

    (d_loss, (variables_g, variables_d)), grad_d = jax.value_and_grad(loss_d, has_aux=True)(
        optimizer_d.target, optimizer_g.target, batch, rng_d, variables_g, variables_d)

    d_loss = jax.lax.pmean(d_loss, axis_name='batch')
    grad_d = jax.lax.pmean(grad_d, axis_name='batch')

    optimizer_d = optimizer_d.apply_gradient(grad_d)

    return rng, variables_g, variables_d, optimizer_g, optimizer_d, d_loss, g_loss


def main(args):
    wandb.init(project='flax-dcgan-selfie')

    dataset = GANDataset({})
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    rng = jax.random.PRNGKey(42)
    rng, rng_g, rng_d = jax.random.split(rng, 3)

    init_batch_g = jnp.ones((1, 1, 1, 100), jnp.float32)
    variables_g = Generator(training=True).init(rng_g, init_batch_g)

    init_batch_d = jnp.ones((1, 32, 32, 3), jnp.float32)
    variables_d = Discriminator(training=True).init(rng_d, init_batch_d)

    optimizer_g = flax.optim.Adam(
        learning_rate=1e-4, beta1=0.5, beta2=0.9).create(variables_g["params"])
    optimizer_g = flax.jax_utils.replicate(optimizer_g)

    optimizer_d = flax.optim.Adam(
        learning_rate=1e-4, beta1=0.5, beta2=0.9).create(variables_d["params"])
    optimizer_d = flax.jax_utils.replicate(optimizer_d)

    variables_g = flax.jax_utils.replicate(variables_g)
    variables_d = flax.jax_utils.replicate(variables_d)

    rngs = jax.random.split(rng, num=jax.local_device_count())

    global_step = 0
    for epoch in range(100):
        for i, (img_a, img_b) in tqdm(enumerate(train_dataloader)):
            img_a = shard(img_a.numpy())
            img_b = shard(img_b.numpy())

            rngs, variables_g, variables_d, optimizer_g, optimizer_d, d_loss, g_loss = train_step(
                rngs, variables_g, variables_d, optimizer_g, optimizer_d, img_a)

            if global_step % 10 == 0:
                to_log = {'g_loss': float(jnp.mean(g_loss)),
                          'd_loss': float(jnp.mean(d_loss))}
                if global_step % 100 == 0:
                    rng, rng_sample = jax.random.split(rng)
                    z = jax.random.normal(rng_sample, shape=(1, 1, 1, 100))

                    temp_params_g = flax.jax_utils.unreplicate(
                        optimizer_g.target)
                    temp_variables_g = flax.jax_utils.unreplicate(variables_g)

                    samples = Generator(training=False).apply(
                        {'params': temp_params_g, 'batch_stats': temp_variables_g['batch_stats']}, z, mutable=False)

                    img = jnp.reshape((samples + 1) / 2, [32, 32, 3])
                    to_log['img'] = wandb.Image(np.array(img))
                wandb.log(to_log)

            global_step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--num_workers', default=0)

    args = parser.parse_args()

    if args.debug:
        import ptvsd
        ptvsd.enable_attach(address=('localhost', 5678),
                            redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()

    main(args)
# @dataclass
# class Data:
#     batch_size: int
#     num_workers: int


# main(Data(256, 4))
