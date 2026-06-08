import tensorflow as tf
from tensorflow.keras.layers import (Activation, Add, BatchNormalization, Conv2D,
                                     Conv2DTranspose, DepthwiseConv2D, Dropout,
                                     Input, Lambda, LayerNormalization,
                                     MaxPooling2D, Multiply, concatenate)


def _directional_scan_average(x):
    """Mix spatial tokens with four directional cumulative scans."""
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]

    h_positions = tf.cast(tf.range(1, height + 1), x.dtype)[tf.newaxis, :, tf.newaxis, tf.newaxis]
    w_positions = tf.cast(tf.range(1, width + 1), x.dtype)[tf.newaxis, tf.newaxis, :, tf.newaxis]

    top_down = tf.cumsum(x, axis=1) / h_positions
    bottom_up = tf.reverse(
        tf.cumsum(tf.reverse(x, axis=[1]), axis=1) / h_positions,
        axis=[1],
    )
    left_right = tf.cumsum(x, axis=2) / w_positions
    right_left = tf.reverse(
        tf.cumsum(tf.reverse(x, axis=[2]), axis=2) / w_positions,
        axis=[2],
    )

    return (top_down + bottom_up + left_right + right_left) * 0.25


def mamba_bottleneck_block(inputs, n_filters, dropout_rate=0.0, name="mamba_bottleneck"):
    """Mamba-inspired 2D residual block for the U-Net bottleneck.

    Swin-UMamba uses VSS/SS2D blocks over image tokens. This TensorFlow block keeps
    the same bottleneck idea without adding PyTorch-only `mamba-ssm` dependencies.
    """
    normalized = LayerNormalization(axis=-1, epsilon=1e-6, name=f"{name}_norm")(inputs)
    projected = Conv2D(n_filters * 2, 1, padding="same", name=f"{name}_in_proj")(normalized)
    spatial = Lambda(lambda x: x[..., :n_filters], name=f"{name}_spatial_split")(projected)
    gate = Lambda(lambda x: x[..., n_filters:], name=f"{name}_gate_split")(projected)

    spatial = DepthwiseConv2D(3, padding="same", name=f"{name}_depthwise")(spatial)
    spatial = Activation("swish", name=f"{name}_spatial_act")(spatial)
    spatial = Lambda(_directional_scan_average, name=f"{name}_scan2d")(spatial)
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
    """Build a U-Net variant with a Mamba-style block only in the bottleneck."""
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
