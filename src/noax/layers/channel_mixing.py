"""Channel mixing layers for neural networks.

This module implements channel mixing layers that operate on multi-channel data.
It provides both linear and MLP-based channel mixing operations.

The implementation follows the principles from:
[1] Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations"

The implementation includes:
1. ChannelMixingLinear: A simple linear transformation across channels
2. ChannelMixingMLP: A multi-layer perceptron operating on channels
"""

import jax
import jax.numpy as jnp
from typing import List, Optional, Callable
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class ChannelMixingLinear(eqx.Module):
    r"""Linear channel mixing layer.

    This layer performs a linear transformation across channels while preserving
    spatial dimensions. It corresponds to W in equation (2) of [1], i.e.
    W : \mathbb{R}^{d_v} \to \mathbb{R}^{d_v}.


    Attributes:
        weights: Linear transformation weights
                 Shape: (in_channels, out_channels)
        bias: Optional bias term for each output channel
              Shape: (out_channels,)
        use_bias: Whether to use bias term
        in_channels: Number of input channels
        out_channels: Number of output channels
        debug: Whether to run in debug mode
    """

    weights: Float[Array, " in_channels out_channels"]
    bias: Optional[Float[Array, " out_channels"]]
    use_bias: bool = eqx.static_field()
    in_channels: int = eqx.static_field()
    out_channels: int = eqx.static_field()
    debug: bool = eqx.static_field()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        key: PRNGKeyArray,
        use_bias: bool = True,
        debug: bool = False,
    ) -> None:
        """Initialize the linear channel mixing layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            key: JAX PRNG key for weight initialization
            use_bias: Whether to use bias term
            debug: Whether to run in debug mode
        """
        self.use_bias = use_bias
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.debug = debug

        key_weights, _ = jax.random.split(key)
        w_shape = (in_channels, out_channels)
        kaiming_scale = jnp.sqrt(2 / in_channels)
        self.weights = kaiming_scale * jax.random.normal(key_weights, w_shape)
        if use_bias:
            self.bias = jnp.zeros(out_channels)
        else:
            self.bias = None

    def __call__(
        self, x: Float[Array, " in_channels *spatial_dims"]
    ) -> Float[Array, " out_channels *spatial_dims"]:
        """Apply linear channel mixing to input.

        Args:
            x: Input tensor of shape (in_channels, *spatial_dims)

        Returns:
            Output tensor of shape (out_channels, *spatial_dims)
            The output maintains the same spatial dimensions as input
            but with transformed channel dimension.

        Raises if debug is True:
            AssertionError: If input dimensions don't match layer dimensions
            AssertionError: If input is not real-valued of 32-bit or 64-bit precision.
        """
        if self.debug:
            assert x.dtype in [
                jnp.float32,
                jnp.float64,
            ], f"Input must be real-valued, got {x.dtype}"
            assert x.shape[0] == self.in_channels, (
                f"Input channels {x.shape[0]} doesn't match expected {self.in_channels}"
            )

        # Contract weights and input over input channels
        out = jnp.einsum("io,i...->o...", self.weights, x)
        if self.use_bias:
            # Add bias term, broadcasting across spatial dimensions
            out = out + self.bias[(...,) + (None,) * (x.ndim - 1)]
        return out


class ChannelMixingMLP(eqx.Module):
    """Multi-layer perceptron for channel mixing.

    This layer implements a multi-layer perceptron that operates on channels
    while preserving spatial dimensions. It consists of multiple linear layers
    with non-linear activations in between.

    Attributes:
        layers: List of linear channel mixing layers
        num_hidden_layers: Number of hidden layers
        in_channels: Number of input channels
        out_channels: Number of output channels
        hidden_channels: Number of channels in hidden layers
        activation: Activation function to use between layers
        debug: Whether to run in debug mode
    """

    layers: List[ChannelMixingLinear]
    num_hidden_layers: int = eqx.static_field()
    in_channels: int = eqx.static_field()
    out_channels: int = eqx.static_field()
    hidden_channels: int = eqx.static_field()
    activation: Callable = eqx.static_field()
    use_bias: bool = eqx.static_field()
    debug: bool = eqx.static_field()

    def __init__(
        self,
        num_hidden_layers: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: Callable,
        key: PRNGKeyArray,
        use_bias: bool = True,
        debug: bool = False,
    ) -> None:
        """Initialize the MLP channel mixing layer.

        Args:
            num_hidden_layers: Number of hidden layers
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Number of channels in hidden layers
            activation: Activation function to use between layers
            key: JAX PRNG key for weight initialization
            use_bias: Whether to use bias terms in linear layers
            debug: Whether to run in debug mode
        """
        self.num_hidden_layers = num_hidden_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.layers = []
        self.activation = activation
        self.use_bias = use_bias
        self.debug = debug

        # Split key for each layer initialization
        in_key, out_key, *hidden_keys = jax.random.split(key, num_hidden_layers + 2)

        # Input layer
        self.layers.append(
            ChannelMixingLinear(in_channels, hidden_channels, in_key, use_bias, debug)
        )

        # Hidden layers
        for i in range(num_hidden_layers):
            self.layers.append(
                ChannelMixingLinear(
                    hidden_channels, hidden_channels, hidden_keys[i], use_bias, debug
                )
            )

        # Output layer
        self.layers.append(
            ChannelMixingLinear(hidden_channels, out_channels, out_key, use_bias, debug)
        )

    def __call__(
        self, x: Float[Array, " in_channels *spatial_dims"]
    ) -> Float[Array, " out_channels *spatial_dims"]:
        r"""Apply MLP channel mixing to input.

        Args:
            x: Input tensor of shape (in_channels, *spatial_dims)

        Returns:
            Output tensor of shape (out_channels, *spatial_dims)
            The output maintains the same spatial dimensions as input
            but with transformed channel dimension.
            It corresponds to P in figure 2.a. of [1], i.e.
            P : \mathbb{R}^{d_a} \to \mathbb{R}^{d_v}.

        Raises if debug is True:
            AssertionError: If input dimensions don't match layer dimensions
            AssertionError: If input is not real-valued of 32-bit or 64-bit precision.
        """
        if self.debug:
            assert x.dtype in [
                jnp.float32,
                jnp.float64,
            ], f"Input must be real-valued, got {x.dtype}"
            assert x.shape[0] == self.in_channels, (
                f"Input channels {x.shape[0]} doesn't match expected {self.in_channels}"
            )

        # Apply all layers except the last one with activation
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # Apply final layer without activation
        return self.layers[-1](x)
