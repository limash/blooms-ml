import os
from pathlib import Path

import jax
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds
from flax.metrics import tensorboard
from flax.training import train_state
from jax import numpy as jnp

from blooms_ml import SimpleMLP, apply_model, update_model
from blooms_ml.configs import default


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds["image"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    model = SimpleMLP(features=[100, 100, 100])
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    train_ds["image"] = jnp.float32(train_ds["image"]) / 255.0
    test_ds["image"] = jnp.float32(test_ds["image"]) / 255.0
    return train_ds, test_ds


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets()
    rng = jax.random.key(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds, config.batch_size, input_rng)
        _, test_loss, test_accuracy = apply_model(state, test_ds["image"], test_ds["label"])

        summary_writer.scalar("train_loss", train_loss, epoch)
        summary_writer.scalar("train_accuracy", train_accuracy, epoch)
        summary_writer.scalar("test_loss", test_loss, epoch)
        summary_writer.scalar("test_accuracy", test_accuracy, epoch)

    summary_writer.flush()
    return state


def main():
    config = default.get_config()
    config.num_epochs = 1
    config.batch_size = 8
    workdir = f"{Path.home()}/blooms-ml_results"
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    train_and_evaluate(config=config, workdir=workdir)


if __name__ == "__main__":
    main()
