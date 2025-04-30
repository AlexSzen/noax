"""Spectral Convolution Module for Neural Networks.

This module implements a spectral convolution layer that operates in the frequency domain.
It uses Fast Fourier Transform (FFT) to convert spatial data to frequency domain, applies
convolution in the frequency domain, then converts back to spatial domain.

The implementation follows the principles from:
[1] Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations"
[2] Kossaifi et al. "Multi-Grid Tensorized Fourier Neural Operator for High Resolution PDEs"
"""

from typing import List, Optional, Tuple, Any
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Complex, PRNGKeyArray


class SpectralConvND:
    """Spectral Convolution Layer for N-dimensional data.

    This layer performs convolution in the frequency domain using Fourier Neural Operator principles.
    The operation consists of:
    1. Converting input to frequency domain using FFT
    2. Applying learned weights to selected frequency modes
    3. Converting back to spatial domain using inverse FFT

    Mathematical Operation:
    Given input x, the operation performs:
    F^{-1}(W · F(x)) where F is the Fourier transform and W are learned weights

    Attributes:
        weights: Complex-valued weights for frequency domain convolution
                 Shape: (in_channels, out_channels, *spatial_dims)
        bias: Optional bias term for each output channel
              Shape: (out_channels, *spatial_dims)
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_modes: Number of Fourier modes to keep in each spatial dimension
        use_bias: Whether to use bias term
        debug: Whether to run in debug mode.
    """

    weights: Complex[Array, " in_channels out_channels *spatial_dims"]
    bias: Optional[Float[Array, " out_channels *spatial_dims"]]
    in_channels: int = eqx.static_field()
    out_channels: int = eqx.static_field()
    n_modes: List[int] = eqx.static_field()
    use_bias: bool = eqx.static_field()
    debug: bool = eqx.static_field()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: List[int],
        use_bias: bool,
        *,
        key: PRNGKeyArray,
        debug: bool = False,
    ) -> None:
        """Initialize the Spectral Convolution layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            n_modes: Number of modes to keep in each spatial dimension
            use_bias: Whether to use bias term
            key: JAX PRNG key for weight initialization
            debug: Whether to run in debug mode.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.use_bias = use_bias
        self.debug = debug
        self._coerce_n_modes(n_modes)

        key_wr, key_wi = jax.random.split(key)

        # Because input is real, we grab 2**(d-1) hyperrectangles
        # at the corners of the input, see [2]. Hence the factor of 2:
        # for each spatial dimension, there are two corners.
        double_modes = (2 * mode for mode in n_modes[:-1])
        w_shape = (in_channels, out_channels, *double_modes, n_modes[-1])
        kaiming_scale_complex = jnp.sqrt(1 / in_channels)
        self.weight = kaiming_scale_complex * (
            jax.random.normal(key_wr, w_shape) + 1j * jax.random.normal(key_wi, w_shape)
        )

        # one bias per output channel, broadcast along spatial dimensions.
        if use_bias:
            b_shape = (out_channels,) + (1,) * len(n_modes)
            self.bias = jnp.zeros(b_shape)
        else:
            self.bias = None

    def _build_slices(self, spatial_shape: Tuple[int, ...]) -> Tuple[slice, ...]:
        """Build slices for extracting frequency modes from input.

        Build slices to grab the corners of the input along the spatial frequency axes.
        All the frequencies are centered on 0, except the last one, which is the real frequency.

        Args:
            spatial_shape: Shape of the spatial frequency dimensions.

        Returns:
            Tuple of slices for extracting the relevant frequency modes from input.
        """
        x_slices = [slice(None)]

        for i in range(len(spatial_shape[:-1])):
            center_slice = spatial_shape[i] // 2
            start_slice = center_slice - self.n_modes[i]
            end_slice = center_slice + self.n_modes[i]
            x_slices += [slice(start_slice, end_slice)]

        x_slices += [slice(0, self.n_modes[-1])]
        return tuple(x_slices)

    def __call__(
        self, x: Float[Array, " in_channels *spatial_dims"]
    ) -> Float[Array, " out_channels *spatial_dims"]:
        """Apply spectral convolution to input.

        Args:
            x: Input tensor of shape (in_channels, *spatial_dims)
            Must be a real-valued array with matching spatial dimensions
            to the layer configuration.

        Returns:
            Output tensor of shape (out_channels, *spatial_dims)
            The output maintains the same spatial dimensions as input
            but with transformed channel dimension.

        Raises if debug is True:
            AssertionError: If input dimensions don't match layer dimensions
            AssertionError: If input is not real-valued
            AssertionError: If spatial dimensions are not at least twice the modes
            AssertionError: If last spatial dimension is not at least as large as the last mode
        """
        _, *spatial_dims = x.shape

        if self.debug:
            assert x.dtype in [
                jnp.float32,
                jnp.float64,
            ], f"Input must be real-valued, got {x.dtype}"
            assert x.shape[0] == self.in_channels, (
                f"Input channels {x.shape[0]} doesn't match expected {self.in_channels}"
            )
            assert len(spatial_dims) == len(self.n_modes), (
                f"Number of spatial dimensions {len(spatial_dims)} does not match "
                f"number of modes {len(self.n_modes)}"
            )
            assert all(
                s > 2 * m for s, m in zip(spatial_dims[:-1], self.n_modes[:-1])
            ), (
                f"Spatial dimensions {spatial_dims} must be at least twice the modes {self.n_modes}"
            )
            assert spatial_dims[-1] >= self.n_modes[-1], (
                f"Last spatial dimension {spatial_dims[-1]} must be at least as large as the last mode {self.n_modes[-1]}"
            )

        # do fft on spatial dimensions
        fft_axes = tuple(range(-len(x.shape) + 1, 0))
        x_ft = jnp.fft.fftshift(jnp.fft.rfftn(x, axes=fft_axes), axes=fft_axes[:-1])

        # create slices to grab corners of input
        slices = self._build_slices(spatial_dims)

        # contract weights and corners of x over input channels.
        out_ft_crop = jnp.einsum("i...,io...->o...", x_ft[slices], self.weight)

        # Create and fill output
        out_ft = jnp.zeros((self.out_channels, *x_ft.shape[1:]), dtype=x_ft.dtype)
        out_ft = out_ft.at[slices].set(out_ft_crop)
        out = jnp.fft.irfftn(
            jnp.fft.fftshift(out_ft, axes=fft_axes[:-1]), axes=fft_axes
        )
        if self.use_bias:
            out += self.bias
        return out

    @staticmethod
    def _coerce_n_modes(n_modes: Any) -> Tuple[int, ...]:
        """Converts and validates n_modes input into a tuple of positive integers.

        Args:
            n_modes (Any): List, tuple, or array-like containing the number of modes per spatial dimension.

        Returns:
            Tuple[int, ...]: A tuple of positive integers, one per spatial dimension.

        Raises:
            TypeError: If `n_modes` is not a list, tuple, or array-like.
            TypeError: If any element in `n_modes` is not an integer.
            ValueError: If `n_modes` is empty or contains non-positive integers.
        """
        if isinstance(n_modes, (np.ndarray, jax.Array)):
            n_modes = tuple(n_modes)
        elif isinstance(n_modes, list):
            n_modes = tuple(n_modes)
        elif isinstance(n_modes, tuple):
            pass
        else:
            raise TypeError(
                f"n_modes must be a list, tuple, or array of integers, but got {type(n_modes)}."
            )

        if len(n_modes) == 0:
            raise ValueError("n_modes must contain at least one element.")
        if not all(isinstance(mode, int) for mode in n_modes):
            raise TypeError("All elements in n_modes must be integers.")
        if not all(mode > 0 for mode in n_modes):
            raise ValueError("All elements in n_modes must be positive.")

        return n_modes
