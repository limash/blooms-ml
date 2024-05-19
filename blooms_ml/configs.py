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

import ml_collections
import optax

from blooms_ml.learning import BinaryClassificator, Regressor
from blooms_ml.networks import MLP
from blooms_ml.utils import (
  get_datasets_classification,
  get_datasets_regression,
)


def classification():

  config = ml_collections.ConfigDict()

  config.get_datasets = get_datasets_classification
  config.trainer = BinaryClassificator

  config.network = MLP
  config.args_network = ml_collections.ConfigDict()
  config.args_network.features = [300, 100, 300, 2]

  config.optimizer = optax.sgd
  config.args_optimizer = ml_collections.ConfigDict()
  config.args_optimizer.learning_rate = 0.1
  config.args_optimizer.momentum = 0.9

  return config


def regression():

  config = ml_collections.ConfigDict()

  config.get_datasets = get_datasets_regression
  config.trainer = Regressor

  config.network = MLP
  config.args_network = ml_collections.ConfigDict()
  config.args_network.features = [300, 100, 300, 1]

  config.optimizer = optax.adam
  config.args_optimizer = ml_collections.ConfigDict()
  config.args_optimizer.learning_rate = 0.01

  return config
