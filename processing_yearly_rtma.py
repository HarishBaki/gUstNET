import xarray as xr
import numpy as np
import pandas as pd
import os, sys, time, glob, re

# set the directory to current directory
root_dir = '/data/harish'
os.chdir(root_dir)

year = sys.argv[1]

files = sorted(glob.glob(f'{root_dir}/intermediate_files/{year}*.nc'))

ds  = xr.open_mfdataset(files,concat_dim='time',combine='nested')
ds = ds.compute()

ds.to_netcdf(f'{root_dir}/rtma_i10fg_NYS_subset/{year}.nc')

