"""Unit tests for FourierLayer class."""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from noax.layers.fourier_layer import FourierLayer
from noax.layers.spectral_conv import SpectralConvND
from noax.layers.channel_mixing import ChannelMixingLinear


@pytest.fixture
def rng_key():
    """Fixture for JAX random key."""
    return jax.random.PRNGKey(42)


def test_fourier_layer_initialization(rng_key):
    """Test initialization of FourierLayer."""
    in_channels = 2
    out_channels = 3
    n_modes = (4, 4)
    use_bias_conv = True
    use_bias_skip = True
    activation = jax.nn.relu

    layer = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias_conv=use_bias_conv,
        use_bias_skip=use_bias_skip,
        activation=activation,
        key=rng_key,
        debug=True,
    )

    assert layer.in_channels == in_channels
    assert layer.out_channels == out_channels
    assert layer.n_modes == n_modes
    assert layer.use_bias_conv == use_bias_conv
    assert layer.use_bias_skip == use_bias_skip
    assert layer.activation is activation
    assert layer.debug is True
    assert isinstance(layer.conv_layer, SpectralConvND)
    assert isinstance(layer.skip_layer, ChannelMixingLinear)


def test_fourier_layer_forward(rng_key):
    """Test forward pass of FourierLayer."""
    in_channels = 2
    out_channels = 3
    n_modes = (4, 4)
    spatial_dims = (32, 32)
    use_bias_conv = False
    use_bias_skip = False

    layer = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias_conv=use_bias_conv,
        use_bias_skip=use_bias_skip,
        activation=jax.nn.relu,
        key=rng_key,
        debug=True,
    )
    x = jnp.ones((in_channels, *spatial_dims))
    y = layer(x)

    assert y.shape == (out_channels, *spatial_dims)
    assert y.dtype == jnp.float32


def test_fourier_layer_input_validation(rng_key):
    """Test input validation for FourierLayer."""
    layer = FourierLayer(
        in_channels=2,
        out_channels=3,
        n_modes=(4, 4),
        use_bias_conv=False,
        use_bias_skip=False,
        activation=jax.nn.relu,
        key=rng_key,
        debug=True,
    )

    with pytest.raises(AssertionError):
        x = jnp.ones((3, 32, 32))  # Wrong number of input channels
        layer(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 32, 32), dtype=jnp.complex64)  # Complex input not allowed
        layer(x)


def test_fourier_layer_output_values(rng_key):
    """Test if output values are reasonable and not all zeros or infinity."""
    in_channels = 1
    out_channels = 1
    n_modes = (4, 4)
    use_bias_conv = False
    use_bias_skip = False

    layer = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias_conv=use_bias_conv,
        use_bias_skip=use_bias_skip,
        activation=jax.nn.relu,
        key=rng_key,
    )
    x = jnp.ones((in_channels, 32, 32))
    y = layer(x)

    assert not jnp.any(jnp.isnan(y))
    assert not jnp.any(jnp.isinf(y))


def test_fourier_layer_bias(rng_key):
    """Test bias functionality in FourierLayer."""
    in_channels = 2
    out_channels = 3
    n_modes = (4, 4)

    layer_with_bias = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias_conv=True,
        use_bias_skip=True,
        activation=jax.nn.relu,
        key=rng_key,
        debug=True,
    )
    layer_without_bias = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias_conv=False,
        use_bias_skip=False,
        activation=jax.nn.relu,
        key=rng_key,
        debug=True,
    )

    x = jnp.zeros((in_channels, 32, 32))
    y_with_bias = layer_with_bias(x)
    y_without_bias = layer_without_bias(x)

    assert y_with_bias.shape == y_without_bias.shape


def test_fourier_layer_spatial_dimensions(rng_key):
    """Test FourierLayer with different spatial dimensions."""
    in_channels = 2
    out_channels = 3
    n_modes_1d = (8,)
    n_modes_2d = (4, 4)
    n_modes_3d = (4, 4, 4)

    # Test 1D spatial dimensions
    layer_1d = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes_1d,
        use_bias_conv=True,
        use_bias_skip=True,
        activation=jax.nn.relu,
        key=rng_key,
        debug=True,
    )
    x_1d = jnp.ones((in_channels, 64))
    y_1d = layer_1d(x_1d)
    assert y_1d.shape == (out_channels, 64)

    # Test 2D spatial dimensions
    layer_2d = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes_2d,
        use_bias_conv=True,
        use_bias_skip=True,
        activation=jax.nn.relu,
        key=rng_key,
        debug=True,
    )
    x_2d = jnp.ones((in_channels, 32, 32))
    y_2d = layer_2d(x_2d)
    assert y_2d.shape == (out_channels, 32, 32)

    # Test 3D spatial dimensions
    layer_3d = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes_3d,
        use_bias_conv=True,
        use_bias_skip=True,
        activation=jax.nn.relu,
        key=rng_key,
        debug=True,
    )
    x_3d = jnp.ones((in_channels, 16, 16, 16))
    y_3d = layer_3d(x_3d)
    assert y_3d.shape == (out_channels, 16, 16, 16)


def test_fourier_layer_different_activations(rng_key):
    """Test FourierLayer with different activation functions."""
    in_channels = 2
    out_channels = 3
    n_modes = (4, 4)
    use_bias_conv = True
    use_bias_skip = True

    # Test with ReLU
    layer_relu = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias_conv=use_bias_conv,
        use_bias_skip=use_bias_skip,
        activation=jax.nn.relu,
        key=rng_key,
        debug=True,
    )
    x = jnp.ones((in_channels, 32, 32))
    y_relu = layer_relu(x)
    assert y_relu.shape == (out_channels, 32, 32)

    # Test with Sigmoid
    layer_sigmoid = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias_conv=use_bias_conv,
        use_bias_skip=use_bias_skip,
        activation=jax.nn.sigmoid,
        key=rng_key,
        debug=True,
    )
    y_sigmoid = layer_sigmoid(x)
    assert y_sigmoid.shape == (out_channels, 32, 32)

    # Test with Tanh
    layer_tanh = FourierLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias_conv=use_bias_conv,
        use_bias_skip=use_bias_skip,
        activation=jax.nn.tanh,
        key=rng_key,
        debug=True,
    )
    y_tanh = layer_tanh(x)
    assert y_tanh.shape == (out_channels, 32, 32)
