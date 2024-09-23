import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, MeanAbsoluteError

# Create model
# Define the basic block
def basic_block(input_layer, n_channels):
    residual = layers.Conv2D(n_channels, 1, padding="same")(input_layer)
    out_layer = layers.Conv2D(n_channels, 3, padding="same")(input_layer)
    out_layer = layers.BatchNormalization()(out_layer)
    out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
    out_layer = layers.Conv2D(n_channels, 3, padding="same")(out_layer)
    out_layer = layers.add([out_layer, residual])
    return out_layer

# Define the down-sampling block
def down_block(input_layer, n_channels):
    residual = layers.Conv2D(n_channels, 1, strides=2, padding="same")(input_layer)
    out_layer = layers.BatchNormalization()(input_layer)
    out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
    out_layer = layers.Conv2D(n_channels, 3, strides=2, padding="same")(out_layer)
    out_layer = layers.BatchNormalization()(out_layer)
    out_layer = layers.LeakyReLU(alpha=0.1)(out_layer)
    out_layer = layers.add([out_layer, residual])
    return out_layer

# Define the U-Net architecture
def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    down1 = down_block(inputs, 64)
    down2 = down_block(down1, 128)
    down3 = down_block(down2, 256)

    # Bottleneck
    bottleneck = basic_block(down3, 512)

    # Decoder
    up1 = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(bottleneck)
    up1 = layers.concatenate([up1, down2])
    up1 = basic_block(up1, 256)

    up2 = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(up1)
    up2 = layers.concatenate([up2, down1])
    up2 = basic_block(up2, 128)

    up3 = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(up2)
    up3 = basic_block(up3, 64)

    # Output layer
    outputs = layers.Conv2D(1, 1)(up3)

    model = models.Model(inputs, outputs)
    return model
