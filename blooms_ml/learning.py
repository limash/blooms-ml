from collections.abc import Sequence

import jax
import optax
from flax import linen as nn
from jax import numpy as jnp


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
