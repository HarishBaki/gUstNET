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


def RTMA_data_splitting(zarr_path,dates_range,in_times,out_times,opt_test=False):
    '''
    Since the RTMA data have gaps in it, we have to the standard reference time-series to extract the rolling window samples.
    '''
    # Define input/output window sizes
    in_times = 3   # Example: 24 input hours (1 day)
    out_times = 1   # Example: 6 output hours (6-hour prediction)
    # create a pandas timetime index for the entire training and validation period
    reference_dates = pd.date_range(start=dates_range[0], end=dates_range[1], freq='h')

    # Define the input and output time windows
    in_steps = pd.Timedelta(hours=in_times)
    out_steps = pd.Timedelta(hours=out_times)

    # create input and output samples by sliding the input window over the entire training and validation period
    in_samples = []
    out_samples = []
    for i in range(len(reference_dates) - in_times - out_times +1):
        in_samples.append(reference_dates[i:i+in_times])
        out_samples.append(reference_dates[i+in_times:i+in_times+out_times])
    in_samples = np.array(in_samples)
    out_samples = np.array(out_samples)
    #print(in_samples.shape, out_samples.shape)

    # Load the RTMA data time-series
    ds = xr.open_zarr(zarr_path)
    time_coord = ds.sel(time=slice(*dates_range)).coords['time']

    original_times = pd.to_datetime(time_coord.values)
    reference_dates = pd.to_datetime(reference_dates)
    # Find missing times by comparing the reference and original time series
    missing_times = reference_dates.difference(original_times)
    #print(f'Missing times: {missing_times}, Total missing times: {len(missing_times)}')
    
    # Filter out in_samples and out_samples that overlap with missing times
    filtered_in_samples = []
    filtered_out_samples = []
    for in_sample, out_sample in zip(in_samples, out_samples):
        # Check if any time in the input or output window is in the missing times
        if any(time in missing_times for time in in_sample) or any(time in missing_times for time in out_sample):
            continue  # Skip this sample if it contains a missing time
        filtered_in_samples.append(in_sample)
        filtered_out_samples.append(out_sample)

    # Convert filtered samples to numpy arrays
    filtered_in_samples = np.array(filtered_in_samples)
    filtered_out_samples = np.array(filtered_out_samples)
    #print(filtered_in_samples.shape, filtered_out_samples.shape)
    
    if not opt_test:
        years = pd.DatetimeIndex(filtered_in_samples[:, 0]).year
        months = pd.DatetimeIndex(filtered_in_samples[:, 0]).month
        validation_samples = np.zeros(len(filtered_in_samples), dtype=bool)
        for year in np.unique(years):
            for month in range(1, 13):
                month_indices = np.where((years == year) & (months == month))[0]
                if len(month_indices) == 0:
                    continue
                # Select a random sample from the month
                if len(month_indices) >= int(6*24):
                    start_index = np.random.choice(len(month_indices) - int(6*24) - 1)
                    validation_indices = month_indices[start_index:start_index + int(6*24)]
                    validation_samples[validation_indices] = True
        
        X_train_times = xr.DataArray(filtered_in_samples[~validation_samples], dims=['sample', 'time_window'])
        Y_train_times = xr.DataArray(filtered_out_samples[~validation_samples],dims=['sample', 'time_window'])
        X_val_times = xr.DataArray(filtered_in_samples[validation_samples],dims=['sample', 'time_window'])
        Y_val_times = xr.DataArray(filtered_out_samples[validation_samples],dims=['sample', 'time_window'])
        #print(X_train_times.shape, Y_train_times.shape, X_val_times.shape, Y_val_times.shape)
        
        return X_train_times, Y_train_times, X_val_times, Y_val_times
    
    else:
        X_test_times = xr.DataArray(filtered_in_samples, dims=['sample', 'time_window'])
        Y_test_times = xr.DataArray(filtered_out_samples, dims=['sample', 'time_window'])
        #print(X_test_times.shape, Y_test_times.shape)
        
        return X_test_times, Y_test_times
    

#----------------------------------------------------------------
# Build base U-net architecture
#----------------------------------------------------------------

def actv_switch(switch, negative_slope):
    '''
    Non-linear Activation switch - 0: linear; 1: non-linear
    If non-linear, then Leaky ReLU Activation function with negative_slope value as input
    '''
    if switch == 0:
        actv = "linear"
    else:
        actv = layers.LeakyReLU(negative_slope=negative_slope)
    return actv

class ReflectPadding2D(Layer):
    '''
    Reflection padding mode. 
    In reflection padding, the padded values are a reflection of the edge values of the input tensor. 
    For example, if the input is [1, 2, 3], reflection padding would give [2, 1, 1, 2, 3, 2].
    This padding is necessary when the filter size is beyond 3. 
    Since the input tensor is convolved with the filter, the filter goes out of bounds of the input tensor at the edges.
    Typically, it will increase the input tensor size by 2*pad_size in each dimension. 
    But after convolution, the output tensor size will be the same as the initial input tensor size.
    '''
    def __init__(self, pad_size, **kwargs):
        self.pad_size = pad_size
        super(ReflectPadding2D, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.pad_size, self.pad_size], [self.pad_size, self.pad_size], [0, 0]], mode='REFLECT')

    def get_config(self):
        config = super(ReflectPadding2D, self).get_config()
        config.update({"pad_size": self.pad_size})
        return config
    
def con2d(input, out_channels, filter, dilation_rate, stride, switch, negative_slope, regularize_value):
    '''
    input: input tensor
    out_channels: number of output feature maps
    filter: filter size
    dilation_rate: dilation rate
    stride: stride
    switch: activation switch
    negative_slope: negative slope for Leaky ReLU
    regularize_value: regularizer factor
    '''
    # Manually apply reflect padding
    pad_size = filter // 2
    # Use ReflectPadding2D instead of custom padding directly
    inp_padded = ReflectPadding2D(pad_size)(input)
    # Apply the convolutional layer with 'valid' padding since we've already padded the input
    return layers.Conv2D(out_channels, (filter, filter), dilation_rate=(dilation_rate, dilation_rate),
                                  strides=stride,
                                  activation=actv_switch(switch, negative_slope),
                                  padding="valid", #padding="same",
                                  use_bias=True,
                                  kernel_regularizer=l2(regularize_value),
                                  bias_regularizer=l2(regularize_value))(inp_padded)

# Residual convolution block
def Res_conv_block(input, out_channels, filter, dilation_rate, stride, switch, negative_slope, regularize_value):
    '''
    input: input tensor
    out_channels: number of output feature maps
    filter: filter size
    dilation_rate: dilation rate
    stride: stride
    switch: activation switch
    negative_slope: negative slope for Leaky ReLU
    regularize_value: regularizer factor
    '''
    y = con2d(input, out_channels, filter, dilation_rate, stride, switch, negative_slope, regularize_value)
    y = con2d(input, out_channels, filter, dilation_rate, stride, switch, negative_slope, regularize_value)
    # Residual connection
    y = layers.Add()([y, con2d(input, out_channels, 1, dilation_rate, stride, switch, negative_slope, regularize_value)])

    return y

# Convolution downsampling block
def Conv_down_block(input, out_channels, filter, dilation_rate, stride, switch, negative_slope, regularize_value):

    # Downsampling using the stride 2
    y = con2d(input, out_channels, filter, dilation_rate, 2, switch, negative_slope, regularize_value)
    y = con2d(y, out_channels, filter, dilation_rate, stride, switch, negative_slope, regularize_value)
    y = con2d(y, out_channels, filter, dilation_rate, stride, switch, negative_slope, regularize_value)

    return y

# Attention block
def Attention(input, num_heads, key_dim):

    layer = layers.MultiHeadAttention(num_heads, key_dim, attention_axes=None)
    y = layer(input, input)

    return y

# Convolution upsampling block using bilinear interpolation
def Conv_up_block(input, out_channels, filter, dilation_rate, stride, switch, negative_slope, regularize_value):

    # Upsampling using the stride 2 with transpose convolution
    y = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(input)
    y = con2d(y, out_channels, filter, dilation_rate, stride, switch, negative_slope, regularize_value)
    y = con2d(y, out_channels, filter, dilation_rate, stride, switch, negative_slope, regularize_value)

    return y

#2D Pooling layer - inputs[0:avg;1:max, input tensor, pool size]
def pool2d(i, input, pool_size):
    if i == 0:
        return layers.AveragePooling2D(pool_size=(pool_size, pool_size),
                                            strides=None,
                                            padding='same',
                                            data_format=None)(input)
    else:
        return layers.MaxPooling2D(pool_size=(pool_size, pool_size),
                                            strides=None,
                                            padding='same',
                                            data_format=None)(input)


