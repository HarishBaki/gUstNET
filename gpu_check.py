# gpu_rank.py

import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Checking if the GPU is available and if it is being used
def run_on_gpu(device_index):
    device_name = f'/device:GPU:{device_index}'
    with tf.device(device_name):
        a = tf.random.uniform([2, 2])
        b = tf.random.uniform([2, 2])
        c = tf.matmul(a, b)
        print(f"Running on {device_name}:")
        print(c.numpy())

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')

# Run the function for each GPU
for i, gpu in enumerate(gpus):
    run_on_gpu(i)
    
