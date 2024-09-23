import xarray as xr
import numpy as np
import zarr
import pandas as pd

import os, sys, time, glob, re

import tensorflow as tf
print(tf.version)
from tensorflow import keras
print(keras.__version__)

from keras import layers
from keras.layers import Layer
from keras import models, losses
from keras.regularizers import l1,l2
from keras.optimizers import Optimizer, Adam
from keras.callbacks import EarlyStopping
strategy = tf.distribute.MirroredStrategy()
# Set the logging level to suppress warnings
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

# set the directory to current directory
root_dir = '/data/harish/'

sys.path.append(root_dir)
from libraries import *

# Generator architecture
def Gen(inp_lat, inp_lon, out_lat, out_lon, chnl, out_vars, filter, dilation_rate, stride, switch, negative_slope, regulazier_value, num_heads, key_dim):
    '''
    inp_lat: input latitude
    inp_lon: input longitude
    out_lat: output latitude
    out_lon: output longitude
    chnl: number of input channels
    out_vars: number of output variables
    filter: filter size
    dilation_rate: dilation rate
    stride: stride
    switch: activation switch
    negative_slope: negative slope for Leaky ReLU
    regulazier_value: regularizer factor
    num_heads: number of attention heads
    key_dim: key dimension
    '''

    input = layers.Input(shape=(inp_lat, inp_lon, chnl))
    #y_static = layers.Input(shape=(out_lat, out_lon, 1))

    #y = layers.Concatenate(axis=-1)([input, y_static])

    y = input

    # Encoding path
    skips = []
    for n_out in [16, 32, 64]:
        y = Res_conv_block(y, n_out, filter, dilation_rate, stride, switch, negative_slope, regulazier_value)
        skips.append(Res_conv_block(y, n_out // 4, filter, dilation_rate, stride, switch, negative_slope, regulazier_value))
        y = Conv_down_block(y, n_out, filter, dilation_rate, stride, switch, negative_slope, regulazier_value)

    # Attention block
    y = Res_conv_block(y, 256, filter, dilation_rate, stride, switch, negative_slope, regulazier_value)
    y = Attention(y, num_heads, key_dim)
    y = Res_conv_block(y, 256, filter, dilation_rate, stride, switch, negative_slope, regulazier_value)
    y = Attention(y, num_heads, key_dim)
    y = Res_conv_block(y, 256, filter, dilation_rate, stride, switch, negative_slope, regulazier_value)

    # Decoding path
    for i, n_out in enumerate([64, 32, 16]):
        y = Conv_up_block(y, n_out, filter, dilation_rate, stride, switch, negative_slope, regulazier_value)
        y = layers.Concatenate(axis=-1)([y, skips[-(i + 1)]])
        y = Res_conv_block(y, n_out, filter, dilation_rate, stride, switch, negative_slope, regulazier_value)

    y = Res_conv_block(y, 32, filter, dilation_rate, stride, switch, negative_slope, regulazier_value)

    y = con2d(y, 8, 1, dilation_rate, stride, 0, 0, regulazier_value)
    y = con2d(y, out_vars, 1, dilation_rate, stride, 0, 0, regulazier_value)

    #return models.Model(inputs=[input,y_static], outputs=y)
    return models.Model(inputs=input, outputs=y)