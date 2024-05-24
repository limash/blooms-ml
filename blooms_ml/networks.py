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

from collections.abc import Sequence

from flax import linen as nn


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f"layers_{i}")(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x


class MLPDropout(nn.Module):
    features: Sequence[int]
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, *args, training:bool, **kwargs):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f"layers_{i}")(x)
            if i != len(self.features) - 1:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
                x = nn.relu(x)
        return x
