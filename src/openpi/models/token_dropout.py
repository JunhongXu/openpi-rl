from __future__ import annotations
from flax.nnx import Module
import jax

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import lax, random

from flax.nnx import rnglib
from flax.nnx.module import Module, first_from
from flax import nnx


class TokenDropout(Module):
    """Create a token dropout layer. Most of the code is adapted from
    https://flax.readthedocs.io/en/latest/_modules/flax/nnx/nn/stochastic.html#Dropout
    Unlike the standard dropoout layer, this layer:
    1. does not scale the inputs by 1 / (1 - rate).
    2. dropout per timestep.
    Like the CFG, it does not scale the outputs.
    """

    def __init__(
        self,
        rng_collection: str = "dropout",
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
    ):
        self.rng_collection = rng_collection

        if isinstance(rngs, rnglib.Rngs):
            self.rngs = rngs[self.rng_collection]
        elif isinstance(rngs, rnglib.RngStream):
            self.rngs = rngs.fork()
        elif rngs is None:
            self.rngs = nnx.data(None)
        else:
            raise TypeError(
                f"rngs must be a Rngs, RngStream or None, but got {type(rngs)}."
            )

    def __call__(
        self,
        inputs,
        dropout_level: jax.Array | None = None,
    ) -> jax.Array:
        """Applies a random dropout mask to the input.

        Args:
          inputs: the inputs that should be randomly masked.
          deterministic: if false the inputs are scaled by ``1 / (1 - rate)`` and
            masked, whereas if true, no mask is applied and the inputs are returned
            as is. The ``deterministic`` flag passed into the call method will take
            precedence over the ``deterministic`` flag passed into the constructor.
          rngs: an optional key, RngStream, or Rngs object used to generate the dropout mask.
            If given it will take precedence over the rngs passed into the constructor.

        Returns:
          The masked inputs reweighted to preserve mean.
        """

        # Each batch has an individual dropout rate.
        key = self.rngs()
        print(key)
        batch_size, T = inputs.shape[:2]
        # The larger the dropout level, the more tokens are dropped.
        mask = jax.random.bernoulli(key, p=dropout_level, shape=(batch_size, T))
        mask = jnp.reshape(mask, (batch_size, T, 1))
        return inputs * mask
