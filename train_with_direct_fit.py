# first thing we do is to set the environment variable CUDA_VISIBLE_DEVICES to the GPU we want to use
import os
#Only CPUs
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Specify the GPUs to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
nGPUs = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
print('Using GPUs:', os.environ["CUDA_VISIBLE_DEVICES"], 'Total:', nGPUs)

import xarray as xr
import numpy as np
import zarr
import os, sys, time, glob, re

import tensorflow as tf
print(tf.version)
import tensorflow.keras as keras
print(keras.__version__)

from keras import layers
from keras.layers import Layer
from keras import models, losses
from keras.regularizers import l1,l2
from keras.optimizers import Optimizer, Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from keras.callbacks import EarlyStopping
strategy = tf.distribute.MirroredStrategy()
# Set the logging level to suppress warnings
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
wandb.init(project="Google-Unet")

# set the directory to current directory
root_dir = '/home/harish/gUstNet'
data_dir = '/data/harish/'
model_dir = '/data/harish/models'

sys.path.append(root_dir)
from libraries import *
from gUstNet_1 import Gen
from Google_UNet import build_unet

#--------------------------------------------#
# --- Loading data ---#
#--------------------------------------------#
zarr_path = f'{data_dir}/rtma_i10fg_NYS_subset.zarr'

# Define input/output window sizes
in_times = 3   # Example: 24 input hours (1 day)
out_times = 1   # Example: 6 output hours (6-hour prediction)

# Extract training, validation and test sample times
train_val_dates_range = ('2021-01-01T00', '2022-12-31T23')
X_train_times, Y_train_times, X_val_times, Y_val_times = RTMA_data_splitting(zarr_path,train_val_dates_range,in_times,out_times,opt_test=False)
test_dates_range = ('2023-01-01T00', '2023-12-31T23')
X_test_times, Y_test_times = RTMA_data_splitting(zarr_path,test_dates_range,in_times,out_times,opt_test=True)

# Extract training, validation and test data using the sample times
ds = xr.open_zarr(zarr_path)
data = ds.i10fg#.transpose(..., 'time')
X_train = data.sel(time=X_train_times).transpose('sample', 'y', 'x','time_window')
Y_train = data.sel(time=Y_train_times).transpose('sample', 'y', 'x','time_window')
X_val = data.sel(time=X_val_times).transpose('sample', 'y', 'x','time_window')
Y_val = data.sel(time=Y_val_times).transpose('sample', 'y', 'x','time_window')
X_test = data.sel(time=X_test_times).transpose('sample', 'y', 'x','time_window')
Y_test = data.sel(time=Y_test_times).transpose('sample', 'y', 'x','time_window')

# Permutate the samples
X_train = X_train[np.random.permutation(X_train.shape[0])]
Y_train = Y_train[np.random.permutation(Y_train.shape[0])]
X_val = X_val[np.random.permutation(X_val.shape[0])]
Y_val = Y_val[np.random.permutation(Y_val.shape[0])]
X_test = X_test[np.random.permutation(X_test.shape[0])]
Y_test = Y_test[np.random.permutation(Y_test.shape[0])]

# Set batch size and compute number of batches
small_batch_size = 16
batch_size = small_batch_size * nGPUs

# To fit the samples equally into the GPUs and batches, we need to exclude the remaining bathces
if len(X_train)%batch_size != 0:
	X_train = X_train[:-int(len(X_train)%batch_size),...]
	Y_train = Y_train[:-int(len(Y_train)%batch_size),...]
if len(X_val)%batch_size != 0:
	X_val = X_val[:-int(len(X_val)%batch_size),...]
	Y_val = Y_val[:-int(len(Y_val)%batch_size),...]
if len(X_test)%batch_size != 0:
	X_test = X_test[:-int(len(X_test)%batch_size),...]
	Y_test = Y_test[:-int(len(Y_test)%batch_size),...]

print('X_train',X_train.shape, 'Y_train',Y_train.shape, 
      'X_val',X_val.shape, 'Y_val',Y_val.shape, 
      'X_test',X_test.shape, 'Y_test',Y_test.shape)

num_train_samples = len(X_train)
num_val_samples = len(X_val)
num_train_batches = num_train_samples // batch_size
num_val_batches = num_val_samples // batch_size

print('Number of training samples:', num_train_samples, 'Number of training batches:', num_train_batches, 
      'Number of validation samples:', num_val_samples, 'Number of validation batches:', num_val_batches)

#--------------------------------------------#
# --- Initialize model ---#
#--------------------------------------------#
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

with strategy.scope():
    generator = build_unet((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    print(generator.summary())
    generator.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=MeanSquaredError(),
        metrics=[RootMeanSquaredError(), MeanAbsoluteError()]
    )

# Now train the model using the shuffled data
generator.fit(
    X_train, Y_train,
    batch_size=batch_size,
    epochs=100,
    callbacks=[early_stopping, WandbMetricsLogger(log_freq=1)],
    validation_data=(X_val, Y_val)
)
generator.save(model_dir+'GoogleUnet_' + str(out_times) + 'HR' + '.keras')

Y_pred = generator.predict(X_test, batch_size=batch_size)
tstLoss, testRMSE, testMAE = generator.evaluate(X_test, Y_test, batch_size=batch_size)
print(f"Test Loss: {tstLoss}")
print(f"Test MAE: {testMAE}")
print(f"Test RMSE: {testRMSE}")