import os
import tempfile
from pathlib import Path

import jax
import ml_collections
import numpy as np
import optax
import pandas as pd
from flax.metrics import tensorboard
from flax.training import train_state
from jax import numpy as jnp

from blooms_ml import SimpleMLP, apply_model, update_model, NUM_CLASSES
from blooms_ml.configs import default


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["observations"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds["observations"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_observations = train_ds["observations"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_observations, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def create_train_state(rng, config, obs_shape):
    """Creates initial `TrainState`."""
    model = SimpleMLP(features=[300, 100, 300, NUM_CLASSES])
    params = model.init(rng, jnp.ones([1, *list(obs_shape[1:])]))["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def get_datasets():
    df = pd.read_parquet(f"{Path.home()}/data_ROHO/300stations-norm.parquet")
    df.rename(columns={'label': 'y'}, inplace=True)
    df['label'] = np.where(df['y'] > 0.1, 1, 0)
    df = df.drop(columns=['P1_netPI', 'P1_c', 'y', 'station', 's_rho'])
    df_train = df[df['ocean_time'] < '2008-01-01']
    df_test = df[df['ocean_time'] > '2008-01-01']
    train_data = {
        'label': df_train['label'].values,
        'observations': jnp.float32(df_train.drop(columns=['ocean_time', 'label']).values),
    }
    test_data = {
        'label': df_test['label'].values,
        'observations': jnp.float32(df_test.drop(columns=['ocean_time', 'label']).values),
    }
    return train_data, test_data


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
    state = create_train_state(init_rng, config, train_ds['observations'].shape)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds, config.batch_size, input_rng)
        _, test_loss, test_accuracy = apply_model(state, test_ds["observations"], test_ds["label"])

        summary_writer.scalar("train_loss", train_loss, epoch)
        summary_writer.scalar("train_accuracy", train_accuracy, epoch)
        summary_writer.scalar("test_loss", test_loss, epoch)
        summary_writer.scalar("test_accuracy", test_accuracy, epoch)

    summary_writer.flush()
    return state


def main():
    config = default.get_config()
    config.num_epochs = 300
    config.batch_size = 100
    workdir = tempfile.mkdtemp(prefix=f"{Path.home()}/blooms-ml_results/")
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    train_and_evaluate(config=config, workdir=workdir)


if __name__ == "__main__":
    main()
