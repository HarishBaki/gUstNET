# gpu_rank.py

import os
import tensorflow as tf


# Get the SLURM rank
rank = int(os.environ['SLURM_PROCID'])
print(f"Rank {rank}: Starting...")
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))

# List available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

print(gpus)

if gpus:
    # Restrict TensorFlow to only use one GPU per process based on rank
    try:
        gpu_to_use = gpus[rank % num_gpus]  # Modulo ensures valid GPU index
        tf.config.experimental.set_visible_devices(gpu_to_use, 'GPU')
        tf.config.experimental.set_memory_growth(gpu_to_use, True)
        print(f"Rank {rank}: Using GPU {gpu_to_use.name}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs are initialized
        print(f"Rank {rank}: {e}")
else:
    print(f"Rank {rank}: No GPUs found.")
'''
