import tensorflow as tf
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv2D,
                                     Conv2DTranspose, DepthwiseConv2D, Dropout,
                                     Input, Lambda, LayerNormalization,
                                     MaxPooling2D, Multiply, concatenate)


@tf.keras.utils.register_keras_serializable(package="mamba")
class SelectiveScan2D(tf.keras.layers.Layer):
    """Selective state-space scan over 2D feature maps (SS2D-style).

    Implements the core Mamba recurrence with input-dependent parameters:

        h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
        y_t = C_t . h_t + D * x_t

    where delta_t (step size), B_t (input gate) and C_t (output gate) are
    projected from the input token itself — the "selective" part of Mamba —
    and A uses the S4D-real initialization A_n = -(n + 1).

    Following VMamba's SS2D, the feature map is flattened into four
    directional token sequences (row-major and column-major, forward and
    reverse). The same SSM weights are shared across directions and the
    four outputs are averaged.
    """

    def __init__(self, state_dim=16, dt_rank=None, **kwargs):
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.dt_rank = dt_rank

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.channels = channels
        if self.dt_rank is None:
            self.dt_rank = max(1, channels // 16)

        # S4D-real initialization: A_n = -(n + 1), stored as log so that
        # A = -exp(A_log) stays negative (stable decay) during training.
        a_init = tf.tile(
            tf.math.log(tf.range(1, self.state_dim + 1, dtype=tf.float32))[tf.newaxis, :],
            [channels, 1],
        )
        self.A_log = self.add_weight(
            name="A_log",
            shape=(channels, self.state_dim),
            initializer=tf.constant_initializer(a_init.numpy()),
            trainable=True,
        )
        self.D = self.add_weight(
            name="D",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )

        # Input-dependent projections: delta uses a low-rank bottleneck as in
        # the reference Mamba implementation; B and C share one projection.
        self.dt_down = tf.keras.layers.Dense(self.dt_rank, use_bias=False, name="dt_down")
        self.dt_up = tf.keras.layers.Dense(
            channels,
            bias_initializer=tf.constant_initializer(0.5413),  # softplus^-1(1.0)
            name="dt_up",
        )
        self.bc_proj = tf.keras.layers.Dense(2 * self.state_dim, use_bias=False, name="bc_proj")
        super().build(input_shape)

    def _selective_scan(self, x):
        """Run the selective SSM along axis 1 of x with shape (B, L, C)."""
        delta = tf.nn.softplus(self.dt_up(self.dt_down(x)))  # (B, L, C)
        bc = self.bc_proj(x)
        b_t = bc[..., : self.state_dim]  # (B, L, N)
        c_t = bc[..., self.state_dim :]  # (B, L, N)

        a = -tf.exp(self.A_log)  # (C, N)
        # Zero-order-hold discretization (with Mamba's simplified delta*B term)
        da = tf.exp(delta[..., tf.newaxis] * a)  # (B, L, C, N)
        dbx = (delta * x)[..., tf.newaxis] * b_t[:, :, tf.newaxis, :]  # (B, L, C, N)

        # Sequential scan, time-major: h_t = da_t * h_{t-1} + dbx_t
        da = tf.transpose(da, [1, 0, 2, 3])
        dbx = tf.transpose(dbx, [1, 0, 2, 3])
        h = tf.scan(
            lambda state, elems: elems[0] * state + elems[1],
            (da, dbx),
            initializer=tf.zeros_like(da[0]),
        )
        h = tf.transpose(h, [1, 0, 2, 3])  # (B, L, C, N)

        y = tf.reduce_sum(h * c_t[:, :, tf.newaxis, :], axis=-1)  # (B, L, C)
        return y + self.D * x

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch, height, width = shape[0], shape[1], shape[2]
        channels = self.channels

        row_major = tf.reshape(inputs, (batch, height * width, channels))
        col_major = tf.reshape(
            tf.transpose(inputs, [0, 2, 1, 3]), (batch, height * width, channels)
        )

        outputs = []
        for sequence in (row_major, col_major):
            outputs.append(self._selective_scan(sequence))
            reversed_seq = tf.reverse(sequence, axis=[1])
            outputs.append(tf.reverse(self._selective_scan(reversed_seq), axis=[1]))

        y_rows = tf.reshape(outputs[0] + outputs[1], (batch, height, width, channels))
        y_cols = tf.transpose(
            tf.reshape(outputs[2] + outputs[3], (batch, width, height, channels)),
            [0, 2, 1, 3],
        )
        return (y_rows + y_cols) * 0.25

    def get_config(self):
        config = super().get_config()
        config.update({"state_dim": self.state_dim, "dt_rank": self.dt_rank})
        return config


def mamba_bottleneck_block(inputs, n_filters, dropout_rate=0.0, state_dim=16, name="mamba_bottleneck"):
    """Mamba residual block for the U-Net bottleneck.

    Mirrors the reference Mamba block: norm -> input projection split into a
    spatial branch and a gate branch -> depthwise conv + selective scan on the
    spatial branch -> SiLU gate -> output projection -> residual.
    """
    normalized = LayerNormalization(axis=-1, epsilon=1e-6, name=f"{name}_norm")(inputs)
    projected = Conv2D(n_filters * 2, 1, padding="same", name=f"{name}_in_proj")(normalized)
    spatial = Lambda(lambda x: x[..., :n_filters], name=f"{name}_spatial_split")(projected)
    gate = Lambda(lambda x: x[..., n_filters:], name=f"{name}_gate_split")(projected)

    spatial = DepthwiseConv2D(3, padding="same", name=f"{name}_depthwise")(spatial)
    spatial = Activation("swish", name=f"{name}_spatial_act")(spatial)
    spatial = SelectiveScan2D(state_dim=state_dim, name=f"{name}_ss2d")(spatial)
    spatial = LayerNormalization(axis=-1, epsilon=1e-6, name=f"{name}_scan_norm")(spatial)

    gate = Activation("swish", name=f"{name}_gate_act")(gate)
    mixed = Multiply(name=f"{name}_gated")([spatial, gate])
    mixed = Conv2D(n_filters, 1, padding="same", name=f"{name}_out_proj")(mixed)
    if dropout_rate > 0:
        mixed = Dropout(dropout_rate, name=f"{name}_dropout")(mixed)

    return Add(name=f"{name}_residual")([inputs, mixed])


def encoder_block(inputs, n_filters=32, dropout_rate=0.3, max_pooling=True):
    """U-Net encoder block copied here to keep the Mamba variant isolated."""
    conv = Conv2D(
        n_filters,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="HeNormal",
    )(inputs)
    conv = Conv2D(
        n_filters,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="HeNormal",
    )(conv)
    conv = BatchNormalization()(conv)

    if dropout_rate > 0:
        conv = Dropout(dropout_rate)(conv)

    next_layer = MaxPooling2D(pool_size=(2, 2))(conv) if max_pooling else conv
    skip_connection = conv
    return next_layer, skip_connection


def decoder_block(prev_layer_input, skip_layer_input, n_filters=32):
    """U-Net decoder block copied here to keep the Mamba variant isolated."""
    up = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding="same")(prev_layer_input)
    merge = concatenate([up, skip_layer_input], axis=3)

    conv = Conv2D(n_filters, 3, activation="relu", padding="same", kernel_initializer="HeNormal")(merge)
    conv = Conv2D(n_filters, 3, activation="relu", padding="same", kernel_initializer="HeNormal")(conv)
    return conv


def build_mamba_unet(input_size=(256, 256, 3), n_filters=32, n_classes=1):
    """Build a U-Net variant with a selective-scan Mamba block in the bottleneck."""
    inputs = Input(input_size)

    e1, s1 = encoder_block(inputs, n_filters, dropout_rate=0, max_pooling=True)
    e2, s2 = encoder_block(e1, n_filters * 2, dropout_rate=0, max_pooling=True)
    e3, s3 = encoder_block(e2, n_filters * 4, dropout_rate=0, max_pooling=True)
    e4, s4 = encoder_block(e3, n_filters * 8, dropout_rate=0.3, max_pooling=True)

    b, _ = encoder_block(e4, n_filters * 16, dropout_rate=0.3, max_pooling=False)
    b = mamba_bottleneck_block(b, n_filters * 16, dropout_rate=0.3)

    d1 = decoder_block(b, s4, n_filters * 8)
    d2 = decoder_block(d1, s3, n_filters * 4)
    d3 = decoder_block(d2, s2, n_filters * 2)
    d4 = decoder_block(d3, s1, n_filters)

    outputs = Conv2D(n_classes, 1, activation="sigmoid", padding="same")(d4)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="mamba_unet")


if __name__ == "__main__":
    model = build_mamba_unet(input_size=(256, 256, 3), n_filters=32, n_classes=1)
    model.summary()
