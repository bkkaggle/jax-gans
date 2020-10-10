import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
import flax

import tensorflow as tf
import tensorflow_datasets as tfds


class Generator(flax.nn.Module):
    def apply(self, z):
        x = flax.nn.Dense(z, features=7*7*64)
        x = jnp.reshape(x, [x.shape[0], 7, 7, 64])
        x = flax.nn.relu(x)
        x = flax.nn.ConvTranspose(
            x, features=32, kernel_size=5, strides=2, padding='SAME')
        x = flax.nn.relu(x)
        x = flax.nn.ConvTranspose(
            x, features=1, kernel_size=5, strides=2, padding='SAME')

        x = jnp.tanh(x)

        return x


class Discriminator(flax.nn.Module):
    def apply(self, x):
        x = flax.nn.Conv(x, features=8, kernel_size=5,
                         strides=2, padding='SAME')
        x = flax.nn.leaky_relu(x, negative_slope=0.2)

        x = flax.nn.Conv(x, features=16, kernel_size=5,
                         strides=1, padding='SAME')
        x = flax.nn.leaky_relu(x, negative_slope=0.2)

        x = flax.nn.Conv(x, features=32, kernel_size=5,
                         strides=2, padding='SAME')
        x = flax.nn.leaky_relu(x, negative_slope=0.2)

        x = flax.nn.Conv(x, features=64, kernel_size=5,
                         strides=1, padding='SAME')
        x = flax.nn.leaky_relu(x, negative_slope=0.2)

        x = flax.nn.Conv(x, features=128, kernel_size=5,
                         strides=2, padding='SAME')
        x = flax.nn.leaky_relu(x, negative_slope=0.2)

        x = jnp.reshape(x.shape[0], 1)

        x = flax.nn.Dense(x, features=2)


def main():
        # Download the data once.
    mnist = tfds.load("mnist")

    def make_dataset(batch_size, seed=1):

        def _preprocess(sample):
            # Convert to floats in [0, 1].
            image = tf.image.convert_image_dtype(sample["image"], tf.float32)
            # Scale the data to [-1, 1] to stabilize training.
            return 2.0 * image - 1.0

        ds = mnist["train"]
        ds = ds.map(map_func=_preprocess,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # ds = ds.cache()
        ds = ds.shuffle(10 * batch_size, seed=seed).repeat().batch(batch_size)
        return iter(tfds.as_numpy(ds))

    # Make the dataset.
    dataset = make_dataset(batch_size=4)

    for step in tqdm(range(1000)):
        batch = next(dataset)


if __name__ == "__main__":
    main()
