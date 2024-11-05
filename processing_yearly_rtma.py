import xarray as xr
import numpy as np
import pandas as pd
import os, sys, time, glob, re

# set the directory to current directory
root_dir = '/data/harish'
os.chdir(root_dir)

variable = sys.argv[1]
year = sys.argv[2]

files = sorted(glob.glob(f'{root_dir}/intermediate_files/{year}*.nc'))

ds  = xr.open_mfdataset(files,concat_dim='time',combine='nested')
ds = ds.compute()

# makedir if not exists
target_dir = f'{root_dir}/rtma_{variable}_NYS_subset'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
ds.to_netcdf(f'{target_dir}/{year}.nc')

