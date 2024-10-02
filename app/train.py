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

import argparse
import os
import tempfile
from pathlib import Path

from blooms_ml import configs
from blooms_ml.learning import train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Prepare ROHO800 modeled data for a ML model.")
    parser.add_argument(
        "--workdir",
        type=str,
        default=os.path.join(Path.home(), "blooms-ml_results/"),
        help="Path to the tensorboard logs.",
    )
    parser.add_argument(
        "--datadir", type=str, default=os.path.join(Path.home(), "data_ROHO/"), help="Path to the input data."
    )
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--save-epochs", default=10, type=int)
    parser.add_argument("--batch-size", default=100000, type=int)
    parser.add_argument("--config", required=True, type=str, choices=("classification", "regression"))

    args = parser.parse_args()

    if args.config == "classification":
        config = configs.classification_model()
    elif args.config == "regression":
        config = configs.regression_model()

    config.num_epochs = args.num_epochs
    config.save_epochs = args.save_epochs
    config.batch_size = args.batch_size

    if not os.path.exists(args.workdir):
        os.mkdir(args.workdir)
    workdir = tempfile.mkdtemp(prefix=args.workdir)
    print(f"Tensorboard log is in {workdir}.")
    datadir = args.datadir

    train_and_evaluate(config=config, workdir=workdir, datadir=datadir)


if __name__ == "__main__":
    main()
