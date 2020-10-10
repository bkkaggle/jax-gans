import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax

import tensorflow as tf
import tensorflow_datasets as tfds


class Generator(flax.nn.Module):
    def apply(self, z, training):
        x = flax.nn.Dense(z, features=7*7*64)
        x = jnp.reshape(x, [x.shape[0], 7, 7, 64])
        x = flax.nn.BatchNorm(
            x, use_running_average=not training, momentum=0.9)
        x = flax.nn.relu(x)
        x = flax.nn.ConvTranspose(
            x, features=32, kernel_size=(5, 5), strides=(2, 2), padding='SAME')
        x = flax.nn.BatchNorm(
            x, use_running_average=not training, momentum=0.9)
        x = flax.nn.relu(x)
        x = flax.nn.ConvTranspose(
            x, features=1, kernel_size=(5, 5), strides=(2, 2), padding='SAME')

        x = jnp.tanh(x)

        return x


class Discriminator(flax.nn.Module):
    def apply(self, x, training):
        x = flax.nn.Conv(x, features=8, kernel_size=(5, 5),
                         strides=(2, 2), padding='SAME')
        x = flax.nn.BatchNorm(
            x, use_running_average=not training, momentum=0.9)
        x = flax.nn.leaky_relu(x, negative_slope=0.2)

        x = flax.nn.Conv(x, features=16, kernel_size=(5, 5),
                         strides=(1, 1), padding='SAME')
        x = flax.nn.BatchNorm(
            x, use_running_average=not training, momentum=0.9)
        x = flax.nn.leaky_relu(x, negative_slope=0.2)

        x = flax.nn.Conv(x, features=32, kernel_size=(5, 5),
                         strides=(2, 2), padding='SAME')
        x = flax.nn.BatchNorm(
            x, use_running_average=not training, momentum=0.9)
        x = flax.nn.leaky_relu(x, negative_slope=0.2)

        x = flax.nn.Conv(x, features=64, kernel_size=(5, 5),
                         strides=(1, 1), padding='SAME')
        x = flax.nn.BatchNorm(
            x, use_running_average=not training, momentum=0.9)
        x = flax.nn.leaky_relu(x, negative_slope=0.2)

        x = flax.nn.Conv(x, features=128, kernel_size=(5, 5),
                         strides=(2, 2), padding='SAME')
        x = flax.nn.BatchNorm(
            x, use_running_average=not training, momentum=0.9)
        x = flax.nn.leaky_relu(x, negative_slope=0.2)

        x = jnp.reshape(x, [x.shape[0], -1])

        x = flax.nn.Dense(x, features=1)

        return x


def make_dataset(batch_size, seed=1):
    mnist = tfds.load("mnist")

    def _preprocess(sample):
        image = tf.image.convert_image_dtype(sample["image"], tf.float32)
        return 2.0 * image - 1.0

    ds = mnist["train"]
    ds = ds.map(map_func=_preprocess,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(10 * batch_size, seed=seed).repeat().batch(batch_size)
    return iter(tfds.as_numpy(ds))


@jax.vmap
def bce_logits_loss(logit, label):
    return jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(-jnp.abs(logit)))


@jax.jit
def loss_g(generator, discriminator, rng, g_state, d_state, batch):
    fake_batch, g_state = sample(
        rng, g_state, generator, batch.shape[0], training=True)

    with flax.nn.stateful(d_state) as d_state:
        fake_logits = discriminator(fake_batch, training=True)

    real_labels = jnp.ones((batch.shape[0],), dtype=jnp.int32)
    return jnp.mean(bce_logits_loss(fake_logits, real_labels)), (g_state, d_state)


@jax.jit
def loss_d(discriminator, generator, rng, g_state, d_state, batch):
    fake_batch, g_state = sample(
        rng, g_state, generator, batch.shape[0], training=True)

    with flax.nn.stateful(d_state) as d_state:
        real_logits = discriminator(batch, training=True)
    with flax.nn.stateful(d_state) as d_state:
        fake_logits = discriminator(fake_batch, training=True)

    real_labels = jnp.ones((batch.shape[0],), dtype=jnp.int32)
    real_loss = bce_logits_loss(real_logits, real_labels)

    fake_labels = jnp.zeros((batch.shape[0],), dtype=jnp.int32)
    fake_loss = bce_logits_loss(fake_logits, fake_labels)

    return jnp.mean(real_loss + fake_loss), (g_state, d_state)


@jax.jit
def train_step(rng, g_state, d_state, optimizer_g, optimizer_d, batch):
    rng, rng_gen, rng_disc = jax.random.split(rng, 3)

    (g_loss, (g_state, d_state)), grad_g = jax.value_and_grad(loss_g, has_aux=True)(
        optimizer_g.target, optimizer_d.target, rng, g_state, d_state, batch)
    optimizer_g = optimizer_g.apply_gradient(grad_g)

    (d_loss, (g_state, d_state)),  grad_d = jax.value_and_grad(loss_d, has_aux=True)(
        optimizer_d.target, optimizer_g.target, rng, g_state, d_state, batch)
    optimizer_d = optimizer_d.apply_gradient(grad_d)

    return optimizer_g, optimizer_d, d_loss, g_loss, rng, g_state, d_state


def sample(rng, state, generator, num_samples, training):
    z = jax.random.normal(rng, shape=(num_samples, 20))

    with flax.nn.stateful(state) as state:
        return generator(z, training=training), state


def main():
    dataset = make_dataset(batch_size=64)

    rng = jax.random.PRNGKey(42)
    rng, rng1 = jax.random.split(rng)

    with flax.nn.stateful() as g_state:
        _, initial_params_g = Generator.init_by_shape(
            rng1, [((1, 20), jnp.float32)], training=True)
        generator = flax.nn.Model(Generator, initial_params_g)

    with flax.nn.stateful() as d_state:
        _, initial_params_d = Discriminator.init_by_shape(
            rng1, [((1, 28, 28, 1), jnp.float32)], training=True)
        discriminator = flax.nn.Model(Discriminator, initial_params_d)

    optimizer_g = flax.optim.Adam(
        learning_rate=1e-4, beta1=0.5, beta2=0.9).create(generator)
    optimizer_d = flax.optim.Adam(
        learning_rate=1e-4, beta1=0.5, beta2=0.9).create(discriminator)

    for i in tqdm(range(20000)):
        optimizer_g, optimizer_d, d_loss, g_loss, rng, g_state, d_state = train_step(
            rng, g_state, d_state, optimizer_g, optimizer_d, next(dataset))

        if i % 1000 == 0:
            samples, g_state = sample(
                rng, g_state, optimizer_g.target, 1, training=False)
            plt.imshow(jnp.reshape((samples + 1) / 2, [28, 28]), cmap='gray')
            plt.show()

        if i % 1000 == 0:
            print(d_loss)
            print(g_loss)


if __name__ == "__main__":
    main()
