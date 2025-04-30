"""Fourier layer implementation.

This module implements a Fourier layer that combines spectral convolution with a skip connection.
The layer performs both frequency domain operations and channel mixing in parallel.

The implementation follows the principles from:
[1] Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations"
"""

import jax
import equinox as eqx
from typing import List, Tuple, Callable
from jaxtyping import Array, Float, PRNGKeyArray
from noax.layers.spectral_conv import SpectralConvND
from noax.layers.channel_mixing import ChannelMixingLinear


class FourierLayer(eqx.Module):
    """Fourier Layer combining spectral convolution with channel mixing.

    This layer implements a Fourier Neural Operator block that consists of:
    1. A spectral convolution branch operating in frequency domain
    2. A linear channel mixing skip connection
    3. An activation function applied to their sum

    The mathematical operation performed is:
    σ(SpectralConv(x) + W(x)) where σ is the activation function and W is a linear transform
    operating pointwise on the spatial dimensions.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_modes: Number of Fourier modes to keep in each spatial dimension
        use_bias_conv: Whether to use bias in spectral convolution
        use_bias_skip: Whether to use bias in skip connection
        activation: Activation function to apply after combining branches
        conv_layer: Spectral convolution layer
        skip_layer: Channel mixing skip connection layer
        debug: Whether to run in debug mode
    """

    in_channels: int = eqx.static_field()
    out_channels: int = eqx.static_field()
    n_modes: Tuple[int] = eqx.static_field()
    use_bias_conv: bool = eqx.static_field()
    use_bias_skip: bool = eqx.static_field()
    activation: Callable = eqx.static_field()
    debug: bool = eqx.static_field()
    conv_layer: SpectralConvND
    skip_layer: ChannelMixingLinear

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        use_bias_conv: bool,
        use_bias_skip: bool,
        activation: Callable,
        *,
        key: PRNGKeyArray,
        debug: bool = False,
    ) -> None:
        """Initialize the Fourier layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_modes: Number of modes to keep in each spatial dimension
            use_bias_conv: Whether to use bias in spectral convolution
            use_bias_skip: Whether to use bias in skip connection
            activation: Activation function to apply after combining branches
            key: JAX PRNG key for weight initialization
            debug: Whether to run in debug mode
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.use_bias_conv = use_bias_conv
        self.use_bias_skip = use_bias_skip
        self.activation = activation
        self.debug = debug

        # Split key for initializing both branches
        key_conv, key_skip = jax.random.split(key)

        # Initialize spectral convolution branch
        self.conv_layer = SpectralConvND(
            in_channels, out_channels, n_modes, use_bias_conv, key=key_conv, debug=debug
        )

        # Initialize skip connection branch
        self.skip_layer = ChannelMixingLinear(
            in_channels, out_channels, use_bias_skip, key=key_skip, debug=debug
        )

    def __call__(
        self, x: Float[Array, " in_channels *spatial_dims"]
    ) -> Float[Array, " out_channels *spatial_dims"]:
        """Apply Fourier layer to input.

        Args:
            x: Input tensor of shape (in_channels, *spatial_dims)
               Must be a real-valued array with matching spatial dimensions.

        Returns:
            Output tensor of shape (out_channels, *spatial_dims)
            The output maintains the same spatial dimensions as input
            but with transformed channel dimension.

        Raises if debug is True:
            AssertionError: If input dimensions don't match layer dimensions
            AssertionError: If input is not real-valued
        """
        # Combine spectral convolution and skip connection outputs
        return self.activation(self.conv_layer(x) + self.skip_layer(x))
