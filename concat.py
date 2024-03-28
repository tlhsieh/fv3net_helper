"""
bash
module load python/3.9
source activate fv3net-prognostic-run
python
"""

import os
import xarray as xr

root = '/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384/simulations'

exprs = ['nudged', 'nudged2', 'nudged3', 'nudged4']
zarrs = ['atmos_dt_atmos', 'diags', 'nudging_tendencies', 'reference_state', 'sfc_dt_atmos', 'state_after_timestep']

os.chdir(root)

if not os.path.exists('combined'):
    os.mkdir('combined')

## combine zarrs
for zarr in zarrs:
    ds = xr.concat([xr.open_zarr(f'{root}/{expr}/{zarr}.zarr') for expr in exprs], dim='time')
    ds.to_zarr(f'{root}/combined/{zarr}.zarr')

for zarr in zarrs:
    ds = xr.open_zarr(f'{root}/combined/{zarr}.zarr')
    print(ds.time)

# ## optional: copy artifacts
# if not os.path.exists('combined/artifacts'):
#     os.mkdir('combined/artifacts')

# for expr in exprs:
#     os.system(f'cp -r {root}/{expr}/artifacts/* {root}/combined/artifacts')

with open(f'{root}/combined/README.txt', 'w') as f:
    f.write('This directory contains the combined output of the following experiments:\n')
    for expr in exprs:
        f.write(f'{expr}\n')

    f.write('\n')
    f.write('The artifact and restart folders are left out intentionally\n')

## rename the combined folder
os.rename('nudged', 'nudged1')
os.rename('combined', 'nudged')