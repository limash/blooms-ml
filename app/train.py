import os
import tempfile
from pathlib import Path

import jax
import ml_collections
import numpy as np
import optax
import pandas as pd
import tqdm.auto as tqdm
from flax.metrics import tensorboard
from flax.training import train_state
from jax import numpy as jnp

from blooms_ml import NUM_CLASSES, SimpleMLP, apply_model, update_model
from blooms_ml.configs import default
from blooms_ml.utils import (
    get_stats,
    labeling,
    timeit,
)


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


@timeit
def get_datasets(datadir):
    # open
    df = pd.read_parquet(f"{datadir}roho800_weekly_average.parquet")
    (p1_c_mean, n1_p_mean, n3_n_mean, n5_s_mean,
     p1_c_std, n1_p_std, n3_n_std, n5_s_std) = get_stats(f"{datadir}cnps_mean_std.csv")
    # label
    df = df.groupby(['station', 's_rho']).apply(labeling, include_groups=False)
    df = df.reset_index().drop(columns='level_2')
    df.rename(columns={'label': 'y'}, inplace=True)
    df['label'] = np.where(df['y'] > 1, 1, 0)
    # clean
    df = df[df['y'].notna()]
    df = df.drop(columns=['station', 's_rho', 'rho', 'y'])
    # "normalize"
    df['P1_c'] = ((df['P1_c'] - float(p1_c_mean)) / float(p1_c_std)).round(2).astype('float32')
    df['N1_p'] = ((df['N1_p'] - float(n1_p_mean)) / float(n1_p_std)).round(2).astype('float32')
    df['N3_n'] = ((df['N3_n'] - float(n3_n_mean)) / float(n3_n_std)).round(2).astype('float32')
    df['N5_s'] = ((df['N5_s'] - float(n5_s_mean)) / float(n5_s_std)).round(2).astype('float32')
    # split
    df_train = df[df['ocean_time'] < '2013-01-01']
    df_test = df[df['ocean_time'] > '2013-01-01']
    del df
    train_data = {
        'label': df_train['label'].values,
        'observations': jnp.float32(df_train.drop(columns=['ocean_time', 'label', 'P1_c']).values),
    }
    test_data = {
        'label': df_test['label'].values,
        'observations': jnp.float32(df_test.drop(columns=['ocean_time', 'label', 'P1_c']).values),
    }
    return train_data, test_data


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str,
                       datadir: str) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets(datadir)
    rng = jax.random.key(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config, train_ds['observations'].shape)

    for epoch in tqdm.tqdm(range(config.num_epochs + 1), desc="Epochs", position=0):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(state, train_ds, config.batch_size, input_rng)
        test_loss, test_accuracy = test_epoch(state, test_ds, config.batch_size)

        summary_writer.scalar("train_loss", train_loss, epoch)
        summary_writer.scalar("train_accuracy", train_accuracy, epoch)
        summary_writer.scalar("test_loss", test_loss, epoch)
        summary_writer.scalar("test_accuracy", test_accuracy, epoch)

    summary_writer.flush()
    return state


def main():
    config = default.get_config()
    config.num_epochs = 10
    config.batch_size = 100000
    workdir = tempfile.mkdtemp(prefix=f"{Path.home()}/blooms-ml_results/")
    print(f"Tensorboard log is in {workdir}.")
    datadir = f"{Path.home()}/data_ROHO/"
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    train_and_evaluate(config=config, workdir=workdir, datadir=datadir)


if __name__ == "__main__":
    main()
