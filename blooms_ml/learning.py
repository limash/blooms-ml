from collections.abc import Sequence

import jax
import ml_collections
import numpy as np
import optax
import tqdm.auto as tqdm
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
from jax import numpy as jnp

from blooms_ml.utils import (
    get_datasets,
)

NUM_CLASSES = 2


class SimpleMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f"layers_{i}")(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x


@jax.jit
def apply_model(state, observations, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, observations)
        one_hot = jax.nn.one_hot(labels, NUM_CLASSES)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["observations"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds["observations"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in tqdm.tqdm(perms, desc="Train batches", position=1, leave=False):
        batch_observations = train_ds["observations"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_observations, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def test_epoch(state, test_ds, batch_size):
    epoch_loss = []
    epoch_accuracy = []

    for i in tqdm.tqdm(range(0, len(test_ds["observations"]), batch_size),
                       desc="Test batches", position=1, leave=False):
        batch_observations = test_ds["observations"][i:i + batch_size, ...]
        batch_labels = test_ds["label"][i:i + batch_size, ...]
        _, loss, accuracy = apply_model(state, batch_observations, batch_labels)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    test_loss = np.mean(epoch_loss)
    test_accuracy = np.mean(epoch_accuracy)
    return test_loss, test_accuracy


def create_train_state(rng, config, obs_shape):
    """Creates initial `TrainState`."""
    model = SimpleMLP(features=[300, 100, 300, NUM_CLASSES])
    params = model.init(rng, jnp.ones([1, *list(obs_shape[1:])]))["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str,
                       datadir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
      datadir: Directory with the input data

    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets(datadir)
    rng = jax.random.key(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config, train_ds['observations'].shape)

    for epoch in tqdm.tqdm(range(config.num_epochs), desc="Epochs", position=0):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds, config.batch_size, input_rng)
        test_loss, test_accuracy = test_epoch(state, test_ds, config.batch_size)

        summary_writer.scalar("train_loss", train_loss, epoch)
        summary_writer.scalar("train_accuracy", train_accuracy, epoch)
        summary_writer.scalar("test_loss", test_loss, epoch)
        summary_writer.scalar("test_accuracy", test_accuracy, epoch)

    summary_writer.flush()
    return state
