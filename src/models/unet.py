import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, BatchNormalization,
                                     Dropout, MaxPooling2D, concatenate, Input)





def encoder_block(inputs, n_filters=32, dropout_rate=0.3, max_pooling=True):
    """
    Bloque de doble convolución con un dropout opcional y max pooling

    """
    conv = Conv2D(n_filters,
     3,
      activation="relu",
       padding="same",
        kernel_initializer="HeNormal")(inputs)

    conv = Conv2D(n_filters,
     3,
      activation="relu",
       padding="same",
        kernel_initializer="HeNormal")(conv)

    conv = BatchNormalization()(conv)

    if dropout_rate > 0:
        conv = Dropout(dropout_rate)(conv)
    
    next_layer = MaxPooling2D(pool_size=(2, 2))(conv) if max_pooling else conv
    skip_connection = conv 
    return next_layer, skip_connection



def decoder_block(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decodificador U-Net
    """
    up = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2),
                         padding="same")(prev_layer_input)

    merge = concatenate([up, skip_layer_input], axis=3)

    conv = Conv2D(n_filters, 3, activation="relu", padding="same",
                  kernel_initializer="HeNormal")(merge)
    conv = Conv2D(n_filters, 3, activation="relu", padding="same",
                  kernel_initializer="HeNormal")(conv)
    return conv

def build_unet(input_size=(256, 256, 3), n_filters=32, n_classes=1):
    """Build and return a U-Net model."""
    inputs = Input(input_size)

    # Encoder — filters double at each level
    e1, s1 = encoder_block(inputs, n_filters, dropout_rate=0, max_pooling=True)
    e2, s2 = encoder_block(e1, n_filters * 2, dropout_rate=0, max_pooling=True)
    e3, s3 = encoder_block(e2, n_filters * 4, dropout_rate=0, max_pooling=True)
    e4, s4 = encoder_block(e3, n_filters * 8, dropout_rate=0.3, max_pooling=True)

    # Bottleneck — no pooling
    b, _ = encoder_block(e4, n_filters * 16, dropout_rate=0.3, max_pooling=False)

    # Decoder — skip connections from encoder
    d1 = decoder_block(b, s4, n_filters * 8)
    d2 = decoder_block(d1, s3, n_filters * 4)
    d3 = decoder_block(d2, s2, n_filters * 2)
    d4 = decoder_block(d3, s1, n_filters)

    # Output
    outputs = Conv2D(n_classes, 1, activation="sigmoid", padding="same")(d4)

    return tf.keras.Model(inputs=inputs, outputs=outputs)



if __name__ == "__main__":
    # Instanciar el modelo con el tamaño del dataset
    unet = build_unet(input_size=(256, 256, 3), n_filters=32, n_classes=1)
    
    # Mostrar la arquitectura y número de parámetros
    unet.summary()
