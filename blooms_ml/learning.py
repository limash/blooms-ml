# Copyright 2024 The Blooms-ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass
from functools import partial

import jax
import ml_collections
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm.auto as tqdm
from flax.metrics import tensorboard
from flax.training import train_state
from jax import numpy as jnp


class TrainState(train_state.TrainState):
    key: jax.Array


@partial(jax.jit, static_argnames=["num_classes"])
def apply_classification_model(state, observations, labels, num_classes):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, observations)
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


def l2_loss(x, alpha):
    return alpha * (x**2).mean()


@partial(jax.jit, static_argnames=["loss_alpha"])
def regression_eval_loss(state, observations, y, loss_alpha=1e-3):
    predictions = jnp.squeeze(
        state.apply_fn(
            {"params": state.params},
            x=observations,
            training=False,
        )
    )
    loss = jnp.mean(optax.losses.l2_loss(predictions=predictions, targets=y))
    loss += sum(l2_loss(w, alpha=loss_alpha) for w in jax.tree_leaves(state.params))
    return loss


@partial(jax.jit, static_argnames=["loss_alpha"])
def apply_regression_model(state, observations, y, loss_alpha=1e-3):
    """Computes gradients, loss and accuracy for a single batch."""

    dropout_train_key = jax.random.fold_in(key=state.key, data=state.step)

    def loss_fn(params):
        predictions = jnp.squeeze(
            state.apply_fn(
                {"params": params},
                x=observations,
                training=True,
                rngs={"dropout": dropout_train_key},
            )
        )
        loss = jnp.mean(optax.losses.l2_loss(predictions=predictions, targets=y))
        loss += sum(l2_loss(w, alpha=loss_alpha) for w in jax.tree_leaves(params))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return grads, loss


@jax.jit
def update_model(state, grads):
    """
    Updates parameters in the network and optimizer, from train_state.apply_gradients:
    Note that internally this function calls ``.tx.update()`` followed by a call
    to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.
    """
    return state.apply_gradients(grads=grads)


@dataclass
class BinaryClassificator:
    NUM_CLASSES: int = 2

    @staticmethod
    def train_epoch(state, train_ds, batch_size, rng):
        """
        Train for a single epoch.
        Returns:
            state: updated flax.training.train_state
            info: dict with auxiliary information
        """

        labels, observations = train_ds["label"], train_ds["observations"]
        coordinates_ones = np.where(labels == 1)[0]
        coordinates_zeros = np.where(labels != 1)[0]
        random_zeros = np.random.choice(coordinates_zeros, len(coordinates_ones), replace=False)
        merged_coordinates = np.concatenate((random_zeros, coordinates_ones))
        train_ds["label"], train_ds["observations"] = labels[merged_coordinates], observations[merged_coordinates]

        train_ds_size = len(train_ds["observations"])
        steps_per_epoch = train_ds_size // batch_size
        # this is fast on gpu and slow on cpu
        perms = jax.random.permutation(rng, len(train_ds["observations"]))
        perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))

        epoch_loss = []
        epoch_accuracy = []

        for perm in tqdm.tqdm(perms, desc="Train batches", position=1, leave=False):
            batch_observations = train_ds["observations"][perm, ...]
            batch_labels = train_ds["label"][perm, ...]
            grads, loss, accuracy = apply_classification_model(
                state, batch_observations, batch_labels, BinaryClassificator.NUM_CLASSES
            )
            state = update_model(state, grads)
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)
        train_loss = np.mean(epoch_loss)
        train_accuracy = np.mean(epoch_accuracy)
        return state, {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }

    @staticmethod
    def test_epoch(state, test_ds, batch_size):
        """
        Returns:
            info: dict with auxiliary information
        """
        epoch_loss = []
        epoch_accuracy = []

        for i in tqdm.tqdm(
            range(0, len(test_ds["observations"]), batch_size), desc="Test batches", position=1, leave=False
        ):
            batch_observations = test_ds["observations"][i : i + batch_size, ...]
            batch_labels = test_ds["label"][i : i + batch_size, ...]
            _, loss, accuracy = apply_classification_model(
                state, batch_observations, batch_labels, BinaryClassificator.NUM_CLASSES
            )
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)
        test_loss = np.mean(epoch_loss)
        test_accuracy = np.mean(epoch_accuracy)
        return {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }


@dataclass
class Regressor:
    @staticmethod
    def train_epoch(state, train_ds, batch_size, rng):
        """
        Train for a single epoch.
        Returns:
            state: updated flax.training.train_state
            info: dict with auxiliary information
        """
        train_ds_size = len(train_ds["observations"])
        steps_per_epoch = train_ds_size // batch_size

        # this is fast on gpu and slow on cpu
        perms = jax.random.permutation(rng, len(train_ds["observations"]))
        perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, batch_size))

        epoch_loss = []

        for perm in tqdm.tqdm(perms, desc="Train batches", position=1, leave=False):
            batch_observations = train_ds["observations"][perm, ...]
            batch_y = train_ds["y"][perm, ...]
            grads, loss = apply_regression_model(state, batch_observations, batch_y)
            state = update_model(state, grads)
            epoch_loss.append(loss)
        train_loss = np.mean(epoch_loss)
        return state, {
            "train_loss": train_loss,
        }

    @staticmethod
    def test_epoch(state, test_ds, batch_size):
        """
        Returns:
            info: dict with auxiliary information
        """
        epoch_loss = []

        for i in tqdm.tqdm(
            range(0, len(test_ds["observations"]), batch_size), desc="Test batches", position=1, leave=False
        ):
            batch_observations = test_ds["observations"][i : i + batch_size, ...]
            batch_y = test_ds["y"][i : i + batch_size, ...]
            loss = regression_eval_loss(state, batch_observations, batch_y)
            epoch_loss.append(loss)
        test_loss = np.mean(epoch_loss)
        return {
            "test_loss": test_loss,
        }


def create_train_state(rng, config, obs_shape):
    """Creates initial `TrainState`."""
    state_rng, init_rng = jax.random.split(rng)
    model = config.network(**config.args_network)
    # Remove 'time' dimension and add 'batch 1' instead
    params = model.init(init_rng, jnp.ones([1, *list(obs_shape[1:])]), training=False)["params"]
    tx = config.optimizer(**config.args_optimizer)
    return TrainState.create(apply_fn=model.apply, params=params, key=state_rng, tx=tx)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str, datadir: str):
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
      datadir: Directory with the input data

    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = config.get_datasets(datadir)

    # orbax so far cannot recognize a new key<fry> dtype, use the old one
    rng = jax.random.PRNGKey(0)  # jax.random.key(0)

    summary_writer = tensorboard.SummaryWriter(os.path.join(workdir, "tensorboard/"))
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config, train_ds["observations"].shape)
    trainer = config.trainer()

    for epoch in tqdm.tqdm(range(1, config.num_epochs + 1), desc="Epochs", position=0):
        rng, input_rng = jax.random.split(rng)
        state, info_train = trainer.train_epoch(state, train_ds, config.batch_size, input_rng)
        info_test = trainer.test_epoch(state, test_ds, config.batch_size)

        info = info_train | info_test
        for key, value in info.items():
            summary_writer.scalar(key, value, epoch)
        summary_writer.flush()

        if epoch % config.save_epochs == 0:
            orbax_checkpointer = ocp.StandardCheckpointer()
            orbax_checkpointer.save(os.path.join(workdir, f"chkpt_epoch_{epoch:03}"), state)
