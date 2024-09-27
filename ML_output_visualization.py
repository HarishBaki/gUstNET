import xarray as xr
import numpy as np
import zarr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, time, glob, re
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature, LAND, COASTLINE

import joblib

# set the directory to current directory
root_dir = '/home/harish/gUstNET'
data_dir = '/data/harish'
temp_dir = f'{root_dir}/tmp'

os.chdir(root_dir)

sys.path.append(root_dir)
from plotters import *

# read the inputs 
# -------------------------------#
model = sys.argv[1]
prediction_horizon = int(sys.argv[2])

zarr_path = f'{data_dir}/rtma_i10fg_NYS_subset.zarr'
ds = xr.open_zarr(zarr_path)

# -------------------------------#
# read npy files
# -------------------------------#
test_file = f'/data/harish/Sukanta_ML_data/ExtractedDATA/GTst_{prediction_horizon}HR'
print(test_file)
tests = np.load(test_file)
tests = tests[:,:,:,0]

predictions_file = f'/data/harish/Sukanta_ML_data/ExtractedDATA/GPred_{model}_{prediction_horizon}HR'
predictions = np.load(predictions_file)
predictions = predictions[:,:,:,0]

# Extract latitude and longitude from the original dataset
latitude = ds.latitude
longitude = ds.longitude
time_dim = ds.time.sel(time=slice('2023-01-01','2023-12-31')).isel(time=slice(prediction_horizon+2,None))

# Create a new xarray dataset
tests_ds = xr.Dataset(
    {
        'data': (('time', 'y', 'x'), tests)  # Add variable with coordinates
    },
    coords={
        'time': time_dim,  # Replace with actual time values if available
        'latitude': (('y', 'x'), latitude.values),  # Add latitude coordinates
        'longitude': (('y', 'x'), longitude.values)  # Add longitude coordinates
    }
)
# Create a new xarray dataset
predictions_ds = xr.Dataset(
    {
        'data': (('time', 'y', 'x'), predictions)  # Add variable with coordinates
    },
    coords={
        'time': time_dim,  # Replace with actual time values if available
        'latitude': (('y', 'x'), latitude.values),  # Add latitude coordinates
        'longitude': (('y', 'x'), longitude.values)  # Add longitude coordinates
    }
)
        
# create a directory with the event_name even if exist
event_dir = f'{temp_dir}/{model}'
os.makedirs(event_dir, exist_ok=True)

def worker (i: int):
    time_slice = pd.to_datetime(time_dim[i].values).strftime('%Y-%m-%dT%H:%M')
    fig = plt.figure(figsize=(8,6), dpi=300)
    gs = matplotlib.gridspec.GridSpec(2, 1)
    map_plotter(fig,gs[0,0],tests_ds.data.isel(time=i),x='longitude',y='latitude',levels=np.arange(0,25.1,1),cmap=turbo_cmap,
                    title=f'Observed Gust at {time_slice}',shrink=0.9,colorbar=True,cbar_label='m/s',
                    orientation='vertical',fontsize=8,pad=0.05)
    map_plotter(fig,gs[1,0],predictions_ds.data.isel(time=i),x='longitude',y='latitude',levels=np.arange(0,25.1,1),cmap=turbo_cmap,
                    title=f'Predicted Gust at {time_slice}',shrink=0.9,colorbar=True,cbar_label='m/s',
                    orientation='vertical',fontsize=8,pad=0.05)
    plt.savefig(f'{event_dir}/{i}.png',bbox_inches='tight',dpi=300)
    plt.close(fig)


# Create a pool of workers
joblib.Parallel(n_jobs=32)(joblib.delayed(worker)(i) for i in range(64))