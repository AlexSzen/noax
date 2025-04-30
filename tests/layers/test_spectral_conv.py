"""Unit tests for SpectralConvND class."""

import pytest
import jax
import jax.numpy as jnp
from noax.layers.spectral_conv import SpectralConvND


@pytest.fixture
def rng_key():
    """Fixture for JAX random key."""
    return jax.random.PRNGKey(42)


# 1D Convolution Tests
def test_spectral_conv_1d_initialization(rng_key):
    """Test initialization of 1D spectral convolution layer."""
    in_channels = 2
    out_channels = 3
    n_modes = [4]
    use_bias = True

    conv = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=use_bias,
        key=rng_key,
    )

    assert conv.in_channels == in_channels
    assert conv.out_channels == out_channels
    assert conv.n_modes == n_modes
    assert conv.use_bias == use_bias
    assert conv.weight.shape == (in_channels, out_channels, 4)
    assert conv.bias.shape == (out_channels, 1)


def test_spectral_conv_1d_forward(rng_key):
    """Test forward pass of 1D spectral convolution layer."""
    in_channels = 2
    out_channels = 3
    length = 64
    n_modes = [16]
    use_bias = False

    conv = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=use_bias,
        key=rng_key,
    )
    x = jnp.ones((in_channels, length))
    y = conv(x)

    assert y.shape == (out_channels, length)
    assert y.dtype == jnp.float32


def test_spectral_conv_1d_input_validation(rng_key):
    """Test input validation for 1D SpectralConvND."""
    conv = SpectralConvND(
        in_channels=2,
        out_channels=3,
        n_modes=[4],
        use_bias=False,
        key=rng_key,
        debug=True,
    )

    with pytest.raises(AssertionError):
        x = jnp.ones((3, 32))  # Wrong number of input channels
        conv(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 32), dtype=jnp.complex64)  # Complex input not allowed
        conv(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 32, 32))  # Wrong number of spatial dimensions (2D vs 1D)
        conv(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 3))  # Spatial dim too small
        conv(x)


def test_spectral_conv_1d_output_values(rng_key):
    """Test if 1D output values are reasonable and not all zeros or infinity."""
    in_channels = 1
    out_channels = 1
    n_modes = [8]
    use_bias = False

    conv = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=use_bias,
        key=rng_key,
        debug=True,
    )
    x = jnp.ones((in_channels, 64))
    y = conv(x)

    assert not jnp.any(jnp.isnan(y))
    assert not jnp.any(jnp.isinf(y))
    assert jnp.any(y != 0)


def test_spectral_conv_1d_bias(rng_key):
    """Test bias functionality in 1D spectral convolution."""
    in_channels = 2
    out_channels = 3
    n_modes = [8]

    conv_with_bias = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=True,
        key=rng_key,
        debug=True,
    )
    conv_without_bias = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=False,
        key=rng_key,
        debug=True,
    )

    x = jnp.zeros((in_channels, 64))
    y_with_bias = conv_with_bias(x)
    y_without_bias = conv_without_bias(x)

    assert conv_with_bias.bias is not None
    assert conv_without_bias.bias is None
    assert y_with_bias.shape == y_without_bias.shape


# 2D Convolution Tests
def test_spectral_conv_2d_initialization(rng_key):
    """Test initialization of 2D spectral convolution layer."""
    in_channels = 2
    out_channels = 3
    n_modes = [4, 4]
    use_bias = True

    conv = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=use_bias,
        key=rng_key,
    )

    assert conv.in_channels == in_channels
    assert conv.out_channels == out_channels
    assert conv.n_modes == n_modes
    assert conv.use_bias == use_bias
    assert conv.weight.shape == (
        in_channels,
        out_channels,
        8,
        4,
    )  # 2*n_modes[0] for first dim
    assert conv.bias.shape == (out_channels, 1, 1)


def test_spectral_conv_2d_forward(rng_key):
    """Test forward pass of 2D spectral convolution layer."""
    in_channels = 2
    out_channels = 3
    h, w = 32, 32
    n_modes = [4, 4]
    use_bias = False

    conv = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=use_bias,
        key=rng_key,
    )
    x = jnp.ones((in_channels, h, w))
    y = conv(x)

    assert y.shape == (out_channels, h, w)
    assert y.dtype == jnp.float32


def test_spectral_conv_2d_input_validation(rng_key):
    """Test input validation for 2D SpectralConvND."""
    conv = SpectralConvND(
        in_channels=2,
        out_channels=3,
        n_modes=[4, 4],
        use_bias=False,
        key=rng_key,
        debug=True,
    )

    with pytest.raises(AssertionError):
        x = jnp.ones((3, 32, 32))  # Wrong number of input channels
        conv(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 32, 32), dtype=jnp.complex64)  # Complex input not allowed
        conv(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 32, 32, 32))  # Wrong number of spatial dimensions (3D vs 2D)
        conv(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 6, 6))  # Spatial dims too small
        conv(x)


def test_spectral_conv_2d_output_values(rng_key):
    """Test if 2D output values are reasonable and not all zeros or infinity."""
    in_channels = 1
    out_channels = 1
    n_modes = [4, 4]
    use_bias = True

    conv = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=use_bias,
        key=rng_key,
    )
    x = jnp.ones((in_channels, 32, 32))
    y = conv(x)

    assert not jnp.any(jnp.isnan(y))
    assert not jnp.any(jnp.isinf(y))
    assert jnp.any(y != 0)


def test_spectral_conv_2d_bias(rng_key):
    """Test bias functionality in 2D spectral convolution."""
    in_channels = 2
    out_channels = 3
    n_modes = [4, 4]

    conv_with_bias = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=True,
        key=rng_key,
    )
    conv_without_bias = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=False,
        key=rng_key,
        debug=True,
    )

    x = jnp.zeros((in_channels, 32, 32))
    y_with_bias = conv_with_bias(x)
    y_without_bias = conv_without_bias(x)

    assert conv_with_bias.bias is not None
    assert conv_without_bias.bias is None
    assert y_with_bias.shape == y_without_bias.shape


# 3D Convolution Tests
def test_spectral_conv_3d_initialization(rng_key):
    """Test initialization of 3D spectral convolution layer."""
    in_channels = 2
    out_channels = 3
    n_modes = [4, 4, 4]
    use_bias = True

    conv = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=use_bias,
        key=rng_key,
    )

    assert conv.in_channels == in_channels
    assert conv.out_channels == out_channels
    assert conv.n_modes == n_modes
    assert conv.use_bias == use_bias
    assert conv.weight.shape == (
        in_channels,
        out_channels,
        8,
        8,
        4,
    )  # 2*n_modes for first two dims
    assert conv.bias.shape == (out_channels, 1, 1, 1)


def test_spectral_conv_3d_forward(rng_key):
    """Test forward pass of 3D spectral convolution layer."""
    in_channels = 2
    out_channels = 3
    d, h, w = 32, 32, 32
    n_modes = [4, 4, 4]
    use_bias = False

    conv = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=use_bias,
        key=rng_key,
    )
    x = jnp.ones((in_channels, d, h, w))
    y = conv(x)

    assert y.shape == (out_channels, d, h, w)
    assert y.dtype == jnp.float32


def test_spectral_conv_3d_input_validation(rng_key):
    """Test input validation for 3D SpectralConvND."""
    conv = SpectralConvND(
        in_channels=2,
        out_channels=3,
        n_modes=[4, 4, 4],
        use_bias=False,
        key=rng_key,
        debug=True,
    )

    with pytest.raises(AssertionError):
        x = jnp.ones((3, 32, 32, 32))  # Wrong number of input channels
        conv(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 16, 16, 16), dtype=jnp.complex64)  # Complex input not allowed
        conv(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 16, 16))  # Wrong number of spatial dimensions (2D vs 3D)
        conv(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 6, 6, 6))  # Spatial dims too small
        conv(x)


def test_spectral_conv_3d_output_values(rng_key):
    """Test if 3D output values are reasonable and not all zeros or infinity."""
    in_channels = 1
    out_channels = 1
    n_modes = [4, 4, 4]
    use_bias = False

    conv = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=use_bias,
        key=rng_key,
    )
    x = jnp.ones((in_channels, 32, 32, 32))
    y = conv(x)

    assert not jnp.any(jnp.isnan(y))
    assert not jnp.any(jnp.isinf(y))
    assert jnp.any(y != 0)


def test_spectral_conv_3d_bias(rng_key):
    """Test bias functionality in 3D spectral convolution."""
    in_channels = 2
    out_channels = 3
    n_modes = [4, 4, 4]

    conv_with_bias = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=True,
        key=rng_key,
    )
    conv_without_bias = SpectralConvND(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        use_bias=False,
        key=rng_key,
        debug=True,
    )

    x = jnp.zeros((in_channels, 32, 32, 32))
    y_with_bias = conv_with_bias(x)
    y_without_bias = conv_without_bias(x)

    assert conv_with_bias.bias is not None
    assert conv_without_bias.bias is None
    assert y_with_bias.shape == y_without_bias.shape


def test_spectral_conv_n_modes_validation():
    """Test validation of n_modes parameter."""
    with pytest.raises(TypeError):
        SpectralConvND._coerce_n_modes(42)

    with pytest.raises(ValueError):
        SpectralConvND._coerce_n_modes([])

    with pytest.raises(TypeError):
        SpectralConvND._coerce_n_modes([1.5, 2])

    with pytest.raises(ValueError):
        SpectralConvND._coerce_n_modes([1, -2])
