"""Fourier Neural Operator model.

This module implements the Fourier Neural Operator model as described in [1].
It consists of a lifting layer which lifts the input to a higher dimensional
channel space, followed by a stack of Fourier layers, and a projection layer
which projects the result to the output channel space.

[1] Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations"
"""

from typing import List, Callable, Tuple
import jax
import equinox as eqx
from jaxtyping import Float, Array, PRNGKeyArray
from noax.layers.fourier_layer import FourierLayer
from noax.layers.channel_mixing import ChannelMixingMLP


class FNO(eqx.Module):
    """Fourier Neural Operator model.

    This model implements a complete Fourier Neural Operator that consists of:
    1. A lifting layer that maps input to a higher-dimensional channel space
    2. A sequence of Fourier layers performing spectral convolutions
    3. A projection layer that maps results to the desired output space

    Attributes:
        layers: List of Fourier layers for spectral processing
        lift_layer: Initial layer to lift input to higher dimensions
        proj_layer: Final layer to project to output dimensions
        num_layers: Number of Fourier layers (static)
        in_channels: Number of input channels (static)
        out_channels: Number of output channels (static)
        hidden_channels: Number of channels in Fourier layers (static)
        activation_fourier: Activation function for Fourier layers (static)
        num_layer_lift: Number of layers in lifting MLP (static)
        hidden_channels_lift: Hidden channels in lifting MLP (static)
        num_layer_proj: Number of layers in projection MLP (static)
        hidden_channels_proj: Hidden channels in projection MLP (static)
        activation_lift: Activation function for lift layer (static)
        activation_proj: Activation function for projection layer (static)
        debug: Whether to run in debug mode (static)
    """

    fourier_layers: List[FourierLayer]
    lift_layer: ChannelMixingMLP
    proj_layer: ChannelMixingMLP
    num_layers: int = eqx.static_field()
    in_channels: int = eqx.static_field()
    out_channels: int = eqx.static_field()
    hidden_channels: int = eqx.static_field()
    n_modes: Tuple[int] = eqx.static_field()
    activation_fourier: Callable = eqx.static_field()
    num_layer_lift: int = eqx.static_field()
    hidden_channels_lift: int = eqx.static_field()
    num_layer_proj: int = eqx.static_field()
    hidden_channels_proj: int = eqx.static_field()
    activation_lift: Callable = eqx.static_field()
    activation_proj: Callable = eqx.static_field()
    debug: bool = eqx.static_field()

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes: Tuple[int],
        activation_fourier: Callable,
        use_bias_fourier: bool,
        num_layer_lift: int,
        hidden_channels_lift: int,
        activation_lift: Callable,
        use_bias_lift: bool,
        num_layer_proj: int,
        hidden_channels_proj: int,
        activation_proj: Callable,
        use_bias_proj: bool,
        *,
        key: PRNGKeyArray,
        debug: bool = False,
    ) -> None:
        """Initialize the FNO model.

        Args:
            num_layers: Number of Fourier layers in the model
            in_channels: Number of input channels
            out_channels: Number of output channels
            hidden_channels: Number of channels in Fourier layers
            n_modes: Number of Fourier modes to keep in each spatial dimension
            activation_fourier: Activation function for Fourier layers
            use_bias_fourier: Whether to use bias in Fourier layers
            num_layer_lift: Number of layers in the lifting MLP
            hidden_channels_lift: Number of hidden channels in lifting MLP
            activation_lift: Activation function for lifting layer
            use_bias_lift: Whether to use bias in lifting layer
            num_layer_proj: Number of layers in projection MLP
            hidden_channels_proj: Number of hidden channels in projection MLP
            activation_proj: Activation function for projection layer
            use_bias_proj: Whether to use bias in projection layer
            key: JAX PRNG key for weight initialization
            debug: Whether to run in debug mode with additional checks

        The initialization process:
        1. Sets up static fields for model configuration
        2. Initializes a sequence of Fourier layers with spectral convolutions
        3. Creates lifting and projection MLPs.
        """

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.activation_fourier = activation_fourier
        self.num_layer_lift = num_layer_lift
        self.hidden_channels_lift = hidden_channels_lift
        self.activation_lift = activation_lift
        self.num_layer_proj = num_layer_proj
        self.hidden_channels_proj = hidden_channels_proj
        self.activation_proj = activation_proj
        self.debug = debug

        key_fourier, key_lift, key_proj = jax.random.split(key, 3)

        # Initialize Fourier layers
        self.fourier_layers = []
        for _ in range(num_layers):
            self.fourier_layers.append(
                FourierLayer(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    n_modes=self.n_modes,
                    use_bias_conv=use_bias_fourier,
                    use_bias_skip=use_bias_fourier,
                    activation=self.activation_fourier,
                    key=key_fourier,
                    debug=self.debug,
                )
            )

        # Initialize lift layer
        self.lift_layer = ChannelMixingMLP(
            num_hidden_layers=self.num_layer_lift,
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.hidden_channels_lift,
            activation=self.activation_lift,
            use_bias=use_bias_lift,
            key=key_lift,
            debug=self.debug,
        )

        # Initialize projection layer
        self.proj_layer = ChannelMixingMLP(
            num_hidden_layers=self.num_layer_proj,
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels_proj,
            activation=self.activation_proj,
            use_bias=use_bias_proj,
            key=key_proj,
            debug=self.debug,
        )

    def __call__(
        self, x: Float[Array, " in_channels *spatial_dims"]
    ) -> Float[Array, " out_channels *spatial_dims"]:
        """Apply the FNO model to the input.

        Args:
            x: Input tensor of shape (in_channels, *spatial_dims)
               Must be a real-valued array with spatial dimensions
               matching the model's configuration.

        Returns:
            Output tensor of shape (out_channels, *spatial_dims)
            The output maintains the same spatial dimensions as input
            but transforms the channel dimension as specified.
        """

        x = self.lift_layer(x)
        for layer in self.fourier_layers:
            x = layer(x)
        return self.proj_layer(x)
