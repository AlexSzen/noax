"""Unit tests for FNO (Fourier Neural Operator) model."""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from noax.models.fno import FNO
import numpy as np


@pytest.fixture
def rng_key():
    """Fixture for JAX random key."""
    return jax.random.PRNGKey(42)


def test_fno_initialization(rng_key):
    """Test initialization of FNO model."""
    num_layers = 4
    in_channels = 2
    out_channels = 3
    hidden_channels = 32
    n_modes = (4, 4)
    activation_fourier = jax.nn.gelu
    activation_lift = jax.nn.relu
    activation_proj = jax.nn.tanh

    model = FNO(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        activation_fourier=activation_fourier,
        use_bias_fourier=True,
        num_layer_lift=2,
        hidden_channels_lift=16,
        activation_lift=activation_lift,
        use_bias_lift=True,
        num_layer_proj=2,
        hidden_channels_proj=16,
        activation_proj=activation_proj,
        use_bias_proj=True,
        key=rng_key,
        debug=True,
    )

    assert model.num_layers == num_layers
    assert model.in_channels == in_channels
    assert model.out_channels == out_channels
    assert model.hidden_channels == hidden_channels
    assert model.activation_fourier is activation_fourier
    assert len(model.layers) == num_layers
    assert isinstance(model.lift_layer, eqx.Module)
    assert isinstance(model.proj_layer, eqx.Module)


def test_fno_forward_1d(rng_key):
    """Test forward pass of FNO model with 1D spatial dimensions."""
    num_layers = 4
    in_channels = 2
    out_channels = 3
    hidden_channels = 32
    n_modes = (8,)
    spatial_dims = (64,)

    model = FNO(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        activation_fourier=jax.nn.gelu,
        use_bias_fourier=True,
        num_layer_lift=2,
        hidden_channels_lift=16,
        activation_lift=jax.nn.relu,
        use_bias_lift=True,
        num_layer_proj=2,
        hidden_channels_proj=16,
        activation_proj=jax.nn.tanh,
        use_bias_proj=True,
        key=rng_key,
    )

    x = jnp.ones((in_channels, *spatial_dims))
    y = model(x)

    assert y.shape == (out_channels, *spatial_dims)
    assert y.dtype == jnp.float32


def test_fno_forward_2d(rng_key):
    """Test forward pass of FNO model with 2D spatial dimensions."""
    num_layers = 4
    in_channels = 2
    out_channels = 3
    hidden_channels = 32
    n_modes = (4, 4)
    spatial_dims = (32, 32)

    model = FNO(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        activation_fourier=jax.nn.gelu,
        use_bias_fourier=True,
        num_layer_lift=2,
        hidden_channels_lift=16,
        activation_lift=jax.nn.relu,
        use_bias_lift=True,
        num_layer_proj=2,
        hidden_channels_proj=16,
        activation_proj=jax.nn.tanh,
        use_bias_proj=True,
        key=rng_key,
    )

    x = jnp.ones((in_channels, *spatial_dims))
    y = model(x)

    assert y.shape == (out_channels, *spatial_dims)
    assert y.dtype == jnp.float32


def test_fno_forward_3d(rng_key):
    """Test forward pass of FNO model with 3D spatial dimensions."""
    num_layers = 4
    in_channels = 2
    out_channels = 3
    hidden_channels = 32
    n_modes = (4, 4, 4)
    spatial_dims = (16, 16, 16)

    model = FNO(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_modes=n_modes,
        activation_fourier=jax.nn.gelu,
        use_bias_fourier=True,
        num_layer_lift=2,
        hidden_channels_lift=16,
        activation_lift=jax.nn.relu,
        use_bias_lift=True,
        num_layer_proj=2,
        hidden_channels_proj=16,
        activation_proj=jax.nn.tanh,
        use_bias_proj=True,
        key=rng_key,
    )

    x = jnp.ones((in_channels, *spatial_dims))
    y = model(x)

    assert y.shape == (out_channels, *spatial_dims)
    assert y.dtype == jnp.float32


def test_fno_input_validation(rng_key):
    """Test input validation for FNO model.

    Note: Most input validation is handled by the underlying layers
    (ChannelMixingMLP and FourierLayer). This test only verifies basic
    shape compatibility between layers.
    """
    model = FNO(
        num_layers=4,
        in_channels=2,
        out_channels=3,
        hidden_channels=32,
        n_modes=(4, 4),
        activation_fourier=jax.nn.gelu,
        use_bias_fourier=True,
        num_layer_lift=2,
        hidden_channels_lift=16,
        activation_lift=jax.nn.relu,
        use_bias_lift=True,
        num_layer_proj=2,
        hidden_channels_proj=16,
        activation_proj=jax.nn.tanh,
        use_bias_proj=True,
        key=rng_key,
        debug=True,
    )

    # Test wrong number of input channels
    with pytest.raises(AssertionError):
        x = jnp.ones((3, 32, 32))  # Wrong number of channels
        model(x)

    # Test mismatched spatial dimensions
    with pytest.raises(AssertionError):
        x = jnp.ones((2, 32, 32, 32))  # 3D input for 2D model
        model(x)


def test_fno_output_values(rng_key):
    """Test if output values are reasonable and not all zeros or infinity."""
    model = FNO(
        num_layers=4,
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        n_modes=(4, 4),
        activation_fourier=jax.nn.gelu,
        use_bias_fourier=True,
        num_layer_lift=2,
        hidden_channels_lift=16,
        activation_lift=jax.nn.gelu,
        use_bias_lift=True,
        num_layer_proj=2,
        hidden_channels_proj=16,
        activation_proj=jax.nn.gelu,
        use_bias_proj=True,
        key=rng_key,
    )

    x = jnp.ones((1, 32, 32))
    y = model(x)

    assert not jnp.any(jnp.isnan(y))
    assert not jnp.any(jnp.isinf(y))
    assert jnp.any(y != 0)


def test_fno_different_activations(rng_key):
    """Test FNO model with different activation function combinations."""
    activations = [jax.nn.relu, jax.nn.gelu, jax.nn.tanh, jax.nn.sigmoid]

    for act in activations:
        model = FNO(
            num_layers=4,
            in_channels=2,
            out_channels=3,
            hidden_channels=32,
            n_modes=(4, 4),
            activation_fourier=act,
            use_bias_fourier=True,
            num_layer_lift=2,
            hidden_channels_lift=16,
            activation_lift=act,
            use_bias_lift=True,
            num_layer_proj=2,
            hidden_channels_proj=16,
            activation_proj=act,
            use_bias_proj=True,
            key=rng_key,
        )

        x = jnp.ones((2, 32, 32))
        y = model(x)
        assert y.shape == (3, 32, 32)
        assert not jnp.any(jnp.isnan(y))
        assert not jnp.any(jnp.isinf(y))


def test_fno_bias_configurations(rng_key):
    """Test FNO model with different bias configurations."""
    # Test with all biases enabled
    model_with_bias = FNO(
        num_layers=4,
        in_channels=2,
        out_channels=3,
        hidden_channels=32,
        n_modes=(4, 4),
        activation_fourier=jax.nn.gelu,
        use_bias_fourier=True,
        num_layer_lift=2,
        hidden_channels_lift=16,
        activation_lift=jax.nn.relu,
        use_bias_lift=True,
        num_layer_proj=2,
        hidden_channels_proj=16,
        activation_proj=jax.nn.tanh,
        use_bias_proj=True,
        key=rng_key,
    )

    # Test with all biases disabled
    model_without_bias = FNO(
        num_layers=4,
        in_channels=2,
        out_channels=3,
        hidden_channels=32,
        n_modes=(4, 4),
        activation_fourier=jax.nn.gelu,
        use_bias_fourier=False,
        num_layer_lift=2,
        hidden_channels_lift=16,
        activation_lift=jax.nn.relu,
        use_bias_lift=False,
        num_layer_proj=2,
        hidden_channels_proj=16,
        activation_proj=jax.nn.tanh,
        use_bias_proj=False,
        key=rng_key,
    )

    x = jnp.ones((2, 32, 32))
    y_with_bias = model_with_bias(x)
    y_without_bias = model_without_bias(x)

    assert y_with_bias.shape == y_without_bias.shape
