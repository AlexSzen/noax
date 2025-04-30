"""Unit tests for ChannelMixingLinear and ChannelMixingMLP classes."""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from noax.layers.channel_mixing import ChannelMixingLinear, ChannelMixingMLP


@pytest.fixture
def rng_key():
    """Fixture for JAX random key."""
    return jax.random.PRNGKey(42)


# ChannelMixingLinear Tests
def test_channel_mixing_linear_initialization(rng_key):
    """Test initialization of ChannelMixingLinear layer."""
    in_channels = 2
    out_channels = 3
    use_bias = True

    layer = ChannelMixingLinear(
        in_channels=in_channels,
        out_channels=out_channels,
        use_bias=use_bias,
        key=rng_key,
        debug=True,
    )

    assert layer.in_channels == in_channels
    assert layer.out_channels == out_channels
    assert layer.use_bias == use_bias
    assert layer.debug == True
    assert layer.weights.shape == (in_channels, out_channels)
    assert layer.bias.shape == (out_channels,)


def test_channel_mixing_linear_forward(rng_key):
    """Test forward pass of ChannelMixingLinear layer."""
    in_channels = 2
    out_channels = 3
    spatial_dims = (32, 32)
    use_bias = False

    layer = ChannelMixingLinear(
        in_channels=in_channels,
        out_channels=out_channels,
        use_bias=use_bias,
        key=rng_key,
        debug=True,
    )
    x = jnp.ones((in_channels, *spatial_dims))
    y = layer(x)

    assert y.shape == (out_channels, *spatial_dims)
    assert y.dtype == jnp.float32


def test_channel_mixing_linear_input_validation(rng_key):
    """Test input validation for ChannelMixingLinear."""
    layer = ChannelMixingLinear(
        in_channels=2, out_channels=3, use_bias=False, key=rng_key, debug=True
    )

    with pytest.raises(AssertionError):
        x = jnp.ones((3, 32, 32))  # Wrong number of input channels
        layer(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 32, 32), dtype=jnp.complex64)  # Complex input not allowed
        layer(x)


def test_channel_mixing_linear_output_values(rng_key):
    """Test if output values are reasonable and not all zeros or infinity."""
    in_channels = 1
    out_channels = 1
    use_bias = False

    layer = ChannelMixingLinear(
        in_channels=in_channels,
        out_channels=out_channels,
        use_bias=use_bias,
        key=rng_key,
    )
    x = jnp.ones((in_channels, 32, 32))
    y = layer(x)

    assert not jnp.any(jnp.isnan(y))
    assert not jnp.any(jnp.isinf(y))
    assert jnp.any(y != 0)


def test_channel_mixing_linear_bias(rng_key):
    """Test bias functionality in ChannelMixingLinear."""
    in_channels = 2
    out_channels = 3

    layer_with_bias = ChannelMixingLinear(
        in_channels=in_channels,
        out_channels=out_channels,
        use_bias=True,
        key=rng_key,
        debug=True,
    )
    layer_without_bias = ChannelMixingLinear(
        in_channels=in_channels,
        out_channels=out_channels,
        use_bias=False,
        key=rng_key,
        debug=True,
    )

    x = jnp.zeros((in_channels, 32, 32))
    y_with_bias = layer_with_bias(x)
    y_without_bias = layer_without_bias(x)

    assert layer_with_bias.bias is not None
    assert layer_without_bias.bias is None
    assert y_with_bias.shape == y_without_bias.shape


def test_channel_mixing_linear_spatial_dimensions(rng_key):
    """Test ChannelMixingLinear with different spatial dimensions."""
    in_channels = 2
    out_channels = 3

    layer = ChannelMixingLinear(
        in_channels=in_channels, out_channels=out_channels, key=rng_key
    )

    # Test 1D spatial dimensions
    x_1d = jnp.ones((in_channels, 64))
    y_1d = layer(x_1d)
    assert y_1d.shape == (out_channels, 64)

    # Test 2D spatial dimensions
    x_2d = jnp.ones((in_channels, 32, 32))
    y_2d = layer(x_2d)
    assert y_2d.shape == (out_channels, 32, 32)

    # Test 3D spatial dimensions
    x_3d = jnp.ones((in_channels, 16, 16, 16))
    y_3d = layer(x_3d)
    assert y_3d.shape == (out_channels, 16, 16, 16)


# ChannelMixingMLP Tests
def test_channel_mixing_mlp_initialization(rng_key):
    """Test initialization of ChannelMixingMLP layer."""
    num_hidden_layers = 2
    in_channels = 2
    out_channels = 3
    hidden_channels = 4
    use_bias = True

    def activation(x):
        return jax.nn.relu(x)

    mlp = ChannelMixingMLP(
        num_hidden_layers=num_hidden_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        activation=activation,
        use_bias=use_bias,
        key=rng_key,
        debug=True,
    )

    assert mlp.num_hidden_layers == num_hidden_layers
    assert mlp.in_channels == in_channels
    assert mlp.out_channels == out_channels
    assert mlp.hidden_channels == hidden_channels
    assert mlp.debug == True
    assert len(mlp.layers) == num_hidden_layers + 2  # input + hidden + output layers
    assert mlp.layers[0].in_channels == in_channels
    assert mlp.layers[0].out_channels == hidden_channels
    assert mlp.layers[-1].in_channels == hidden_channels
    assert mlp.layers[-1].out_channels == out_channels
    # Check debug flag is passed to all layers
    assert all(layer.debug == True for layer in mlp.layers)


def test_channel_mixing_mlp_forward(rng_key):
    """Test forward pass of ChannelMixingMLP layer."""
    num_hidden_layers = 2
    in_channels = 2
    out_channels = 3
    hidden_channels = 4
    spatial_dims = (32, 32)
    use_bias = False

    def activation(x):
        return jax.nn.relu(x)

    mlp = ChannelMixingMLP(
        num_hidden_layers=num_hidden_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        activation=activation,
        use_bias=use_bias,
        key=rng_key,
        debug=True,
    )
    x = jnp.ones((in_channels, *spatial_dims))
    y = mlp(x)

    assert y.shape == (out_channels, *spatial_dims)
    assert y.dtype == jnp.float32


def test_channel_mixing_mlp_input_validation(rng_key):
    """Test input validation for ChannelMixingMLP."""

    def activation(x):
        return jax.nn.relu(x)

    mlp = ChannelMixingMLP(
        num_hidden_layers=2,
        in_channels=2,
        out_channels=3,
        hidden_channels=4,
        activation=activation,
        use_bias=False,
        key=rng_key,
        debug=True,
    )

    with pytest.raises(AssertionError):
        x = jnp.ones((3, 32, 32))  # Wrong number of input channels
        mlp(x)

    with pytest.raises(AssertionError):
        x = jnp.ones((2, 32, 32), dtype=jnp.complex64)  # Complex input not allowed
        mlp(x)


def test_channel_mixing_mlp_output_values(rng_key):
    """Test if output values are reasonable and not all zeros or infinity."""
    num_hidden_layers = 2
    in_channels = 1
    out_channels = 1
    hidden_channels = 4
    use_bias = True

    def activation(x):
        return jax.nn.relu(x)

    mlp = ChannelMixingMLP(
        num_hidden_layers=num_hidden_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        activation=activation,
        use_bias=use_bias,
        key=rng_key,
    )
    x = jnp.ones((in_channels, 32, 32))
    y = mlp(x)

    assert not jnp.any(jnp.isnan(y))
    assert not jnp.any(jnp.isinf(y))
    assert jnp.any(y != 0)


def test_channel_mixing_mlp_bias(rng_key):
    """Test bias functionality in ChannelMixingMLP."""
    num_hidden_layers = 2
    in_channels = 2
    out_channels = 3
    hidden_channels = 4

    def activation(x):
        return jax.nn.relu(x)

    mlp_with_bias = ChannelMixingMLP(
        num_hidden_layers=num_hidden_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        activation=activation,
        use_bias=True,
        key=rng_key,
        debug=True,
    )
    mlp_without_bias = ChannelMixingMLP(
        num_hidden_layers=num_hidden_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        activation=activation,
        use_bias=False,
        key=rng_key,
        debug=True,
    )

    x = jnp.zeros((in_channels, 32, 32))
    y_with_bias = mlp_with_bias(x)
    y_without_bias = mlp_without_bias(x)

    assert all(layer.bias is not None for layer in mlp_with_bias.layers)
    assert all(layer.bias is None for layer in mlp_without_bias.layers)
    assert y_with_bias.shape == y_without_bias.shape


def test_channel_mixing_mlp_spatial_dimensions(rng_key):
    """Test ChannelMixingMLP with different spatial dimensions."""
    num_hidden_layers = 2
    in_channels = 2
    out_channels = 3
    hidden_channels = 4
    use_bias = True

    def activation(x):
        return jax.nn.relu(x)

    mlp = ChannelMixingMLP(
        num_hidden_layers=num_hidden_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        activation=activation,
        use_bias=use_bias,
        key=rng_key,
        debug=True,
    )

    # Test 1D spatial dimensions
    x_1d = jnp.ones((in_channels, 64))
    y_1d = mlp(x_1d)
    assert y_1d.shape == (out_channels, 64)

    # Test 2D spatial dimensions
    x_2d = jnp.ones((in_channels, 32, 32))
    y_2d = mlp(x_2d)
    assert y_2d.shape == (out_channels, 32, 32)

    # Test 3D spatial dimensions
    x_3d = jnp.ones((in_channels, 16, 16, 16))
    y_3d = mlp(x_3d)
    assert y_3d.shape == (out_channels, 16, 16, 16)


def test_channel_mixing_mlp_different_activations(rng_key):
    """Test ChannelMixingMLP with different activation functions."""
    num_hidden_layers = 2
    in_channels = 2
    out_channels = 3
    hidden_channels = 4
    use_bias = True

    # Test with ReLU
    mlp_relu = ChannelMixingMLP(
        num_hidden_layers=num_hidden_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        activation=jax.nn.relu,
        use_bias=use_bias,
        key=rng_key,
        debug=True,
    )
    x = jnp.ones((in_channels, 32, 32))
    y_relu = mlp_relu(x)
    assert y_relu.shape == (out_channels, 32, 32)

    # Test with Sigmoid
    mlp_sigmoid = ChannelMixingMLP(
        num_hidden_layers=num_hidden_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        activation=jax.nn.sigmoid,
        use_bias=use_bias,
        key=rng_key,
        debug=True,
    )
    y_sigmoid = mlp_sigmoid(x)
    assert y_sigmoid.shape == (out_channels, 32, 32)

    # Test with Tanh
    mlp_tanh = ChannelMixingMLP(
        num_hidden_layers=num_hidden_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        activation=jax.nn.tanh,
        use_bias=use_bias,
        key=rng_key,
        debug=True,
    )
    y_tanh = mlp_tanh(x)
    assert y_tanh.shape == (out_channels, 32, 32)
