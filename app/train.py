#!/usr/bin/env python3

import argparse
import os
import tempfile
from pathlib import Path

from blooms_ml.configs import default
from blooms_ml.learning import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description='Prepare ROHO800 modeled data for a ML model.')
    parser.add_argument(
        "--workdir",
        type=str,
        default=f"{Path.home()}/blooms-ml_results/",
        help="Path to the tensorboard logs."
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default=f"{Path.home()}/data_ROHO/",
        help="Path to the input data."
    )
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=100000, type=int)

    args = parser.parse_args()

    config = default.get_config()
    config.num_epochs = args.num_epochs
    config.batch_size = args.batch_size

    if not os.path.exists(args.workdir):
        os.mkdir(args.workdir)
    workdir = tempfile.mkdtemp(prefix=args.workdir)
    print(f"Tensorboard log is in {workdir}.")
    datadir = args.datadir

    train_and_evaluate(config=config, workdir=workdir, datadir=datadir)


if __name__ == "__main__":
    main()
