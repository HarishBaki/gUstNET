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
root_dir = '/home/harish/gUstNet'
data_dir = '/data/harish/'

sys.path.append(root_dir)
from libraries import *
from gUstNet_1 import Gen
from Google_UNet import build_unet

#--------------------------------------------#
# --- Loading data ---#
#--------------------------------------------#
zarr_path = f'{data_dir}/rtma_i10fg_NYS_subset.zarr'
train_val_dates_range = ('2021-01-01T00', '2022-12-31T23')
# Define input/output window sizes
in_times = 3   # Example: 24 input hours (1 day)
out_times = 1   # Example: 6 output hours (6-hour prediction)
X_train_times, Y_train_times, X_val_times, Y_val_times = RTMA_data_splitting(zarr_path,train_val_dates_range,in_times,out_times,opt_test=False)

test_dates_range = ('2023-01-01T00', '2023-12-31T23')
X_test_times, Y_test_times = RTMA_data_splitting(zarr_path,test_dates_range,in_times,out_times,opt_test=True)

ds = xr.open_zarr(zarr_path)
data = ds.i10fg#.transpose(..., 'time')
X_train = data.sel(time=X_train_times).transpose('sample', 'y', 'x','time_window')
Y_train = data.sel(time=Y_train_times).transpose('sample', 'y', 'x','time_window')
X_val = data.sel(time=X_val_times).transpose('sample', 'y', 'x','time_window')
Y_val = data.sel(time=Y_val_times).transpose('sample', 'y', 'x','time_window')
X_test = data.sel(time=X_test_times).transpose('sample', 'y', 'x','time_window')
Y_test = data.sel(time=Y_test_times).transpose('sample', 'y', 'x','time_window')

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

# --------------------------------------------------------------
# Batches data load
# --------------------------------------------------------------
def batch_data_load(var, start, end):
    #return tf.convert_to_tensor(var.sel(sample=slice(start,end)).values, dtype=tf.float32)
    return var.sel(sample=slice(start,end)).values

#--------------------------------------------#
# --- Initialize model ---#
#--------------------------------------------#
# Define model and compile it within the strategy scope
with strategy.scope():
	#generator = Gen(X_train.shape[1], X_train.shape[2], Y_train.shape[1], Y_train.shape[2], 
	#	          X_train.shape[3], Y_train.shape[3], 5, 1, 1, 1, 0.2, 0, 1, 64)
    generator = build_unet((X_train.shape[1], X_train.shape[2], X_train.shape[3])) 
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.MeanAbsoluteError()
	# Compile the model
    generator.compile(optimizer=optimizer, loss='mean_absolute_error', 
                      metrics=['mean_squared_error', 'mean_absolute_error'])
    print(generator.summary())

# Function to compute loss and gradients
#@tf.function
def train_step(X_batch, Y_batch):
    with tf.GradientTape() as tape:
        Y_pred = generator(X_batch, training=True)
        loss = loss_fn(Y_batch, Y_pred)
    gradients = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss

# Validation step
#@tf.function
def val_step(X_batch, Y_batch):
    predictions = generator(X_batch, training=False)
    loss = tf.keras.losses.MeanAbsoluteError()(Y_batch, predictions)
    return loss

#--------------------------------------------#
# --- Train model ---#
#--------------------------------------------#
train_loss_history = []
val_loss_history = []
# Early stopping parameters
patience = 10  # Number of epochs with no improvement before stopping
best_val_loss = float('inf')
epochs_without_improvement = 0

# Manual training loop
for epoch in range(100):  # Number of epochs
    print(f"Epoch {epoch + 1}/{100}")
    epoch_train_loss = 0
    epoch_val_loss = 0

    # Shuffle the dataset indices
    indices = np.random.permutation(num_train_samples)
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]

    for batch_idx in range(num_train_batches):
        # Load batch data using your custom batch_data_load function
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        X_batch = batch_data_load(X_train_shuffled, start_idx, end_idx)
        Y_batch = batch_data_load(Y_train_shuffled, start_idx, end_idx)
        
        print(f"Batch {batch_idx}: X_batch shape: {X_batch.shape}, Y_batch shape: {Y_batch.shape}")

        # Distribute batch across GPUs and train
        per_replica_losses = strategy.run(train_step, args=(X_batch, Y_batch))
        
        # Reduce losses from all GPUs
        batch_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        epoch_train_loss += batch_loss
        

    # Compute average training loss for the epoch
    avg_train_loss = epoch_train_loss / num_train_batches
    train_loss_history.append(avg_train_loss)
    print(f"Training loss: {avg_train_loss:.4f}")
    
	# Validation loop
    val_loss = 0
    for val_batch_idx in range(num_val_batches):
        # Load validation batch
        start_val_idx = val_batch_idx * batch_size
        end_val_idx = (val_batch_idx + 1) * batch_size
        X_val_batch = batch_data_load(X_val, start_val_idx, end_val_idx)
        Y_val_batch = batch_data_load(Y_val, start_val_idx, end_val_idx)

        # Distribute validation batch across GPUs and evaluate
        per_replica_val_losses = strategy.run(val_step, args=(X_val_batch, Y_val_batch))
        
        # Reduce validation losses
        batch_val_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_val_losses, axis=None)
        val_loss += batch_val_loss

    avg_val_loss = val_loss / num_val_batches
    val_loss_history.append(avg_val_loss)
    print(f"Validation loss: {avg_val_loss:.4f}")
    
	# Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        print("Validation loss improved, saving the model...")
        generator.save(f'{root_dir}/gUstNet_best_model.h5')
    else:
        epochs_without_improvement += 1
        print(f"No improvement in validation loss for {epochs_without_improvement} epochs.")

    # Check for early stopping
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

    print("-" * 50)
    
#--------------------------------------------#
# --- Evaluate model ---#
#--------------------------------------------#
results = generator.evaluate(X_test, Y_test, batch_size=batch_size)
print('test loss, test acc:', results)
