#----------------------------------------
# Import packages
#----------------------------------------
import sys
import os
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["NCCL_DEBUG"] = "WARN"
import time
import socket
import math
import numpy as np
import xarray as xr
import dask
import dask.array as da
import zarr as zr
import pickle
import tensorflow as tf
print(tf.version)
from tensorflow import keras
import horovod.tensorflow as hvd
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import models, losses
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.optimizers import Optimizer
# Set the logging level to suppress warnings
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)
#------------------------------------------------------------
# Intialise Horovod
# -----------------------------------------------------------
hvd.init()
print ('***hvd.size ', hvd.size(),' hvd.rank', hvd.rank(), 'hvd.local_rank() ', hvd.local_rank())
# Horovod: pin GPU to be used to process local rank (one GPU per process)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
print(' gpus = ', gpus)
if hvd.local_rank() == 0:
    print("Socket and len gpus = ",socket.gethostname(), len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
#------------------------------------------------------------- 
# load input data from zarr
#-------------------------------------------------------------
def load_data(data, split):
    out = xr.open_zarr(f'/g/data/xv83/jp6794/EtoB/Data/{data}-{split}.zarr')
    var = f'{data}-{split}'
    if data == 'B2':
        # Selecting the SEA domain
        return out[var].sel(variable = ['tas']).sel(lat=slice(-20, -44.6), lon=slice(130, 154.6))
    else:
        # Selecting the TAS domain
        #return out[var].sel(latitude=slice(-39, -45.75), longitude=slice(143, 149.75))
        return out[var].sel(latitude=slice(-36.75, -46), longitude=slice(142, 151.25))
# Load the static input data from zarr
x_static = xr.open_zarr('/g/data/xv83/jp6794/EtoB/Data/B_C2_static.zarr')['BC2_static']
#---------------------------------------------------------
Epoch_size = load_data('E5', 'train').shape[0]
Val_size  = load_data('E5', 'test').shape[0]
# Batch size - aim to fill GPU memory to achieve best computational performance
batch_size = 32

if hvd.rank() == 0:
    print ('*** rank = ', hvd.rank(),' Epoch size = ', Epoch_size)
    print ('*** rank = ', hvd.rank(),' Val_size = ', Val_size)
    print ('*** rank = ', hvd.rank(),' Batch size = ', batch_size)
#----------------------------------------------------------------
# Horovod: Split the training and val data across multiple processors   
#----------------------------------------------------------------
def hvd_data_split_idx(rank, sample_size, size):
    istart = int(rank*sample_size/size)
    istop  = int((rank+1)*sample_size/size)
    return istart, istop
istart, istop = hvd_data_split_idx(hvd.rank(), Epoch_size, hvd.size())
i_val_start, i_val_stop = hvd_data_split_idx(hvd.rank(), Val_size, hvd.size())

print ( '*** rank = ', hvd.rank(),' istart = ', istart, ' istop = ', istop)
print ( '*** rank = ', hvd.rank(),' i_val_start = ', i_val_start, ' i_val_stop = ', i_val_stop)

#----------------------------------------------------------------
# Horovod: shuffle the training and val data across multiple processors   
#----------------------------------------------------------------
# Set the same seed for all GPUs to shuffle data consistently
seed = 1
# Function to shuffle data indices
def shuffle_data_indices(indices, seed):
    np.random.seed(seed)
    np.random.shuffle(indices)
    return indices

train_indices = shuffle_data_indices(list(range(Epoch_size)), seed)[istart:istop]
val_indices = shuffle_data_indices(list(range(Val_size)), seed)[i_val_start:i_val_stop]

#-----------------------------------------------------------------
# Horovod: Read in the train and val data across multiple processors
#-----------------------------------------------------------------
load_data_start = time.time()

b2c = xr.open_zarr('/g/data/xv83/jp6794/EtoB/Data/BARRA_C2_pr_data_tas.zarr')
y_train = b2c['pr'].sel(time=slice('1980-01-01T00:00:00.000000000', 
                          '2013-12-29T15:30:00.000000000')).expand_dims(channel=1, axis=-1)[train_indices,...]

y_val = b2c['pr'].sel(time=slice('2014-01-01T00:00:00.000000000', 
                          '2022-12-31T15:30:00.000000000')).expand_dims(channel=1, axis=-1)[val_indices,...]
x_train = load_data('E5', 'train')[train_indices,...]
x_val = load_data('E5', 'test')[val_indices,...]

load_data_end = time.time()

if hvd.rank() == 2:
    print('*** rank = ', hvd.rank(), 'time for load_data fn: ',load_data_end - load_data_start)

print('*** rank = ', hvd.rank(),' Training data shapes = ', x_train.shape, y_train.shape)
print('*** rank = ', hvd.rank(),' Val data shapes = ', x_val.shape, y_val.shape)

# Determine how many batches are there in train and val sets
train_batches = int(math.floor(len(train_indices) / batch_size))
val_batches = int(math.floor(len(val_indices) / batch_size))

print ('*** rank = ', hvd.rank(),' train_batches', train_batches)
print ('*** rank = ', hvd.rank(),' val_batches', val_batches)

#----------------------------------------------------------------
# Build base U-net architecture
#----------------------------------------------------------------
# Non-linear Activation switch - 0: linear; 1: non-linear
# If non-linear, then Leaky ReLU Activation function with alpha value as input
def actv_swtch(swtch, alpha_val):
    if swtch == 0:
        actv = "linear"
    else:
        actv = layers.LeakyReLU(alpha=alpha_val)
    return actv

class ReflectPadding2D(Layer):
    def __init__(self, pad_size, **kwargs):
        self.pad_size = pad_size
        super(ReflectPadding2D, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.pad_size, self.pad_size], [self.pad_size, self.pad_size], [0, 0]], mode='REFLECT')

    def get_config(self):
        config = super(ReflectPadding2D, self).get_config()
        config.update({"pad_size": self.pad_size})
        return config
# Defining a generic 3D convolution layer for our use
# Inputs: [input tensor, output feature maps, filter size, dilation rate, stride,
# activation switch, if actv switch is 1 then activation-LReLU-alpha value, 
# regularizer factor]
def con2d(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):
    # Manually apply reflect padding
    pad_size = fil // 2
    # Use ReflectPadding2D instead of custom padding directly
    inp_padded = ReflectPadding2D(pad_size)(inp)
    # Apply the convolutional layer with 'valid' padding since we've already padded the input
    return layers.Conv2D(n_out, (fil, fil), dilation_rate=(dil_rate, dil_rate),
                                  strides=std,
                                  activation=actv_swtch(swtch, alpha_val),
                                  padding="valid", #padding="same",
                                  use_bias=True,
                                  kernel_regularizer=l2(reg_val),
                                  bias_regularizer=l2(reg_val))(inp_padded)

# Residual convolution block
def Res_conv_block(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):

    y = con2d(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    # Residual connection
    y = layers.Add()([y, con2d(inp, n_out, 1, dil_rate, std, swtch, alpha_val, reg_val)])

    return y

# Convolution downsampling block
def Conv_down_block(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):

    # Downsampling using the stride 2
    y = con2d(inp, n_out, fil, dil_rate, 2, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)

    return y

# Attention block
def Attention(inp, num_heads, key_dim):

    layer = layers.MultiHeadAttention(num_heads, key_dim, attention_axes=None)
    y = layer(inp, inp)

    return y

# Convolution upsampling block using bilinear interpolation
def Conv_up_block(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):

    # Upsampling using the stride 2 with transpose convolution
    y = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(inp)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)

    return y

#2D Pooling layer - inputs[0:avg;1:max, input tensor, pool size]
def pool2d(i, inp, ps):
    if i == 0:
        return layers.AveragePooling2D(pool_size=(ps, ps),
                                            strides=None,
                                            padding='same',
                                            data_format=None)(inp)
    else:
        return layers.MaxPooling2D(pool_size=(ps, ps),
                                            strides=None,
                                            padding='same',
                                            data_format=None)(inp)


# Generator architecture
def Gen(inp_lat, inp_lon, out_lat, out_lon, chnl, out_vars, fil, dil_rate, std, swtch, alpha_val, reg_val, num_heads, key_dim):

    inp = layers.Input(shape=(inp_lat, inp_lon, chnl))
    y_st = layers.Input(shape=(out_lat, out_lon, 2))

    # Interpolate the input to target shape using bilinear interpolation
    y = tf.image.resize(inp, [out_lat, out_lon], method='bilinear')

    y = layers.Concatenate(axis=-1)([y, y_st])

    # Encoding path
    skips = []
    for n_out in [64, 128, 256]:
        y = Res_conv_block(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
        skips.append(Res_conv_block(y, n_out // 4, fil, dil_rate, std, swtch, alpha_val, reg_val))
        y = Conv_down_block(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)

    # Attention block
    y = Res_conv_block(y, 256, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = Attention(y, num_heads, key_dim)
    y = Res_conv_block(y, 256, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = Attention(y, num_heads, key_dim)
    y = Res_conv_block(y, 256, fil, dil_rate, std, swtch, alpha_val, reg_val)

    # Decoding path
    for i, n_out in enumerate([256, 128, 64]):
        y = Conv_up_block(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
        y = layers.Concatenate(axis=-1)([y, skips[-(i + 1)]])
        y = Res_conv_block(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)

    y = Res_conv_block(y, 32, fil, dil_rate, std, swtch, alpha_val, reg_val)

    y = con2d(y, 32, 1, dil_rate, std, 0, 0, reg_val)
    y = con2d(y, out_vars, 1, dil_rate, std, 0, 0, reg_val)

    return models.Model(inputs=[inp, y_st], outputs=y)
#----------------------------------------------------------------
# Defining loss functions
#----------------------------------------------------------------
MAE = tf.keras.losses.MeanAbsoluteError()
MSE = tf.keras.losses.MeanSquaredError()
# --------------------------------------------------------------
# Batches data load
# --------------------------------------------------------------
def batch_data_load(var, start, end):
    return tf.convert_to_tensor(var[start:end,...].compute().values, dtype=tf.float32)
# -------------------------------------------------------------------------
# Model initialising and summary print 
# -------------------------------------------------------------------------
# Build the generator model
generator = Gen(x_train.shape[1], x_train.shape[2], y_train.shape[1], y_train.shape[2], 
                  x_train.shape[3], y_train.shape[3], 5, 1, 1, 1, 0.2, 0, 1, 64)
print(generator.summary()) if hvd.rank() == 1 else None
g_opt = tf.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)

'''
generator = tf.keras.models.load_model('/g/data/xv83/jp6794/EtoB/Runs/B2C_tas_pr/CGAN_Run1/Gen_CGAN_pr_21.h5',custom_objects={'ReflectPadding2D': ReflectPadding2D})
print(generator.summary()) if hvd.rank() == 1 else None
g_opt = tf.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-07, decay=0.0, amsgrad=False)
'''
#---------------------------------------------------------------------------
# Train the model
#---------------------------------------------------------------------------
#--------------------------------------------------------------------
# Define first training step (defined seperately to effectively utilise tf.function),
# training step and validation step
#--------------------------------------------------------------------
@tf.function
def first_training_step(x, y_st, y):

    # Train the model
    with tf.GradientTape() as g_tape:
       
        # Forward pass
        y_pr = generator([x, y_st], training=True)
        MSE_loss = MSE(y, y_pr)

    # Horovod: add Horovod Distributed GradientTape
    g_tape = hvd.DistributedGradientTape(g_tape)
    # Compute gradients
    g_gradients = g_tape.gradient(MSE_loss, generator.trainable_variables)
    # Update weights
    g_opt.apply_gradients(zip(g_gradients, generator.trainable_variables))

    hvd.broadcast_variables(generator.variables, root_rank=0)
    hvd.broadcast_variables(g_opt.variables(), root_rank=0)

    return MSE_loss 

@tf.function
def training_step(x, y_st, y):
    # Train the model
    with tf.GradientTape() as g_tape:
       
        # Forward pass
        y_pr = generator([x, y_st], training=True)
        MSE_loss = MSE(y, y_pr)

    # Horovod: add Horovod Distributed GradientTape
    g_tape = hvd.DistributedGradientTape(g_tape)
    # Compute gradients
    g_gradients = g_tape.gradient(MSE_loss, generator.trainable_variables)
    # Update weights
    g_opt.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return MSE_loss 
# ------------------------------------------------------------------------
# Add a barrier to sync all processes before starting training
barrier = hvd.allreduce(tf.constant(0))
print ('*** rank = ', hvd.rank(),' Train model')
#----------------------------------------------------------------
# Running epochs
#----------------------------------------------------------------
max_epochs = 100
previous_epoch = 0
for epoch in range(previous_epoch, max_epochs):
    epoch_start = time.time()
    lr = 1e-5
    if hvd.rank() == 2: 
        print(f"Epoch {epoch+1}: Learning Rate = {lr}")
    g_opt.learning_rate.assign(lr)
    #------------------------------------------------------------------------------------------------
    # Horovod: shuffle the training and val data across multiple processors
    # Set the same seed for all GPUs to shuffle data consistently
    # Randomising the train and val data in each epoch and training and testing on 200 and 60 batches
    # -----------------------------------------------------------------------------------------------
    train_indices = shuffle_data_indices(list(range(Epoch_size)), epoch)[istart:istop]
    val_indices = shuffle_data_indices(list(range(Val_size)), epoch)[i_val_start:i_val_stop]

    y_train = b2c['pr'].sel(time=slice('1980-01-01T00:00:00.000000000', 
                          '2013-12-29T15:30:00.000000000'))[train_indices,...]
    y_val = b2c['pr'].sel(time=slice('2014-01-01T00:00:00.000000000', 
                          '2022-12-31T15:30:00.000000000'))[val_indices,...]
    x_train = load_data('E5', 'train')[train_indices,...]
    x_val = load_data('E5', 'test')[val_indices,...]
    #-----------------------------------
    # Executing training step on batches
    # ----------------------------------
    jstart = 0
    for batch in range(train_batches):
        jend = jstart + batch_size
        if jend >= len(train_indices):
            jend = len(train_indices)

        x_in = batch_data_load(x_train, jstart, jend)
        y_out = batch_data_load(y_train, jstart, jend)
        xs_in = batch_data_load(x_static, 0, batch_size)

        if ((epoch == 0) and (batch == 0)):
            mse_batch_loss = first_training_step(x_in, xs_in, y_out[..., tf.newaxis])
        else:
            mse_batch_loss = training_step(x_in, xs_in, y_out[..., tf.newaxis])

        jstart = jend

        if ((hvd.rank() == 1) & (batch % 50 == 0)):
            print(f"rank:{hvd.rank()}, epoch:{epoch+1}, batch no:{batch}, mse_batch_loss:{mse_batch_loss}")

    epoch_train_end = time.time()
    if hvd.rank() == 2:
        print('*** rank = ', hvd.rank(), 'Lap time for epoch - training: ', epoch_train_end - epoch_start)
    if hvd.rank() == 0:  
        generator.save('/g/data/xv83/jp6794/EtoB/Runs/B2C_tas_pr/Unet_Runs/Base_model_run/Unet_base_{}.h5'.format(epoch+1))