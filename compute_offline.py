"""
bash
module load python/3.9
source activate fv3net-image

Complete settings in nudge-to-fine-stellar-workflow/configs/machine-learning/tq-testing-data_predict.yaml
"""

##### from workflows/diagnostics/fv3net/diagnostics/offline/compute.py
import argparse
import json
import logging
import os
import sys
from tempfile import NamedTemporaryFile
from typing import List, Sequence, Tuple

import yaml

import dataclasses
import fsspec
import fv3fit
import loaders
import vcm
import xarray as xr
from toolz import compose_left
from vcm import interpolate_to_pressure_levels, safe

import intake

INPUT_SENSITIVITY = "input_sensitivity.png"
DIAGS_NC_NAME = "offline_diagnostics.nc"
TRANSECT_NC_NAME = "transect_lon0.nc"
METRICS_JSON_NAME = "scalar_metrics.json"
METADATA_JSON_NAME = "metadata.json"

DERIVATION_DIM_NAME = "derivation"
DELP = "pressure_thickness_of_atmospheric_layer"
PREDICT_COORD = "predict"
TARGET_COORD = "target"
########################################

##### a hack to make the code run
from fv3net.diagnostics.offline.compute import get_prediction, load_grid_info, _compute_diagnostics, plot_input_sensitivity, _get_data_mapper_if_exists, select_snapshot, is_3d, insert_column_integrated_vars, _get_transect, _write_nc, _add_derived_diagnostics

root = '/ncrc/home2/Tsung-Lin.Hsieh/nudge-to-fine-stellar-workflow'
expr_root = '/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384'
res = 'c384'

# root = '/home/hsiehtl/nudge-to-fine-stellar-workflow'
# expr_root = '/scratch/cimes/hsiehtl/C48'
# res = 'c48'

model_tag = '_sfc1000_no0202-04_seed1'
# model_tag = '_subsampleRatio0p015625_balanced'
# model_tag = '_subsampleRatio0p015625_balanceddQ2'
# model_tag = '_subsampleRatio0p015625_balanceddQ2v2'
# model_tag = '_subsampleRatio0p015625_sfc100'
# model_tag = '_subsampleRatio0p015625_sfc500'
# model_tag = '_subsampleRatio0p015625'
# model_tag = '_wna'
# model_tag = '_wna_seed1'
# model_tag = '_wus'
# model_tag = '_subsampleRatio0p0009765625_balanced'
# model_tag = '_subsampleRatio0p0009765625'

class args: # this is a hack to replace the parser
    evaluation_grid = res
    data_yaml = f'{root}/configs/machine-learning/tq-testing-data_predict.yaml'
    catalog_path = f'{root}/configs/catalog.yaml'
    model_path = f'{expr_root}/models{model_tag}/tq-model'
    output_path = f'{expr_root}/offline/test/tq-model'
    n_jobs = -1
    snapshot_time = None
########################################
    
##### from workflows/diagnostics/fv3net/diagnostics/offline/compute.py
# logger.info("Starting diagnostics routine.")

with fsspec.open(args.data_yaml, "r") as f:
    as_dict = yaml.safe_load(f)
config = loaders.BatchesLoader.from_dict(as_dict)
catalog = intake.open_catalog(args.catalog_path)
evaluation_grid = load_grid_info(catalog, args.evaluation_grid)

# logger.info("Opening ML model")
model = fv3fit.load(args.model_path)

# add Q2 and total water path for PW-Q2 scatterplots and net precip domain averages
if any(["Q2" in v for v in model.output_variables]):
    model = fv3fit.DerivedModel(model, derived_output_variables=["Q2"])
gsrm = vcm.gsrm_name_from_resolution_string(args.evaluation_grid)
if gsrm == "fv3gfs":
    horizontal_dims = ["x", "y", "tile"]
elif gsrm == "scream":
    horizontal_dims = ["ncol"]
ds_predicted = get_prediction(
    config=config,
    model=model,
    catalog=catalog,
    evaluation_resolution=evaluation_grid.sizes[horizontal_dims[0]],
)

output_data_yaml = os.path.join(args.output_path, "data_config.yaml")
with fsspec.open(args.data_yaml, "r") as f_in, fsspec.open(
    output_data_yaml, "w"
) as f_out:
    f_out.write(f_in.read())

# compute diags
ds_diagnostics, ds_scalar_metrics = _compute_diagnostics(
    ds_predicted,
    evaluation_grid,
    predicted_vars=model.output_variables,
    n_jobs=args.n_jobs,
)

ds_diagnostics = ds_diagnostics.update(evaluation_grid)
########################################

##### from def _compute_diagnostics()
ds = ds_predicted
predicted_vars = model.output_variables

diagnostic_vars_3d = [var for var in predicted_vars if is_3d(ds[var])]
ds = ds.pipe(insert_column_integrated_vars, diagnostic_vars_3d).load()

full_predicted_vars = [var for var in ds if DERIVATION_DIM_NAME in ds[var].dims]
if "dQ2" in full_predicted_vars or "Q2" in full_predicted_vars:
    full_predicted_vars.append("water_vapor_path")
prediction = safe.get_variables(
    ds.sel({DERIVATION_DIM_NAME: PREDICT_COORD}), full_predicted_vars
)
target = safe.get_variables(
    ds.sel({DERIVATION_DIM_NAME: TARGET_COORD}), full_predicted_vars
)

# prediction.to_netcdf("prediction.nc")
# target.to_netcdf("target.nc")
########################################

##### plot maps
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from helper import coarse_grain, latlon, crop

def box(da):
    return latlon(da, lat, lon, lims)

ds_land = xr.open_zarr(f'{expr_root}/simulations/nudged/sfc_dt_atmos.zarr') # for land_mask

# itile = 2; lims = [42, 52, -127, -117] # PNW # to compare the ratio between two mountains
itile = 2; lims = [42, 62, -135, -115] # PNW+
# itile = 4; lims = [34, 42, -124, -117] # CA-NV # to compare the first 5 days
# itile = 4; lims = [12, 42, -134, -104] # CA-NV+

if res == 'c48':
    lat = ds_land['lat'].isel(tile=itile).values
    lon = ds_land['lon'].isel(tile=itile).values - 360
else:
    lat = ds_land['lat'].isel(tile=itile).isel(time=0).values
    lon = ds_land['lon'].isel(tile=itile).isel(time=0).values - 360

land_mask = ds_land['SLMSKsfc'].isel(tile=itile).isel(time=0)

# field = 'column_integrated_dQ1'
field = 'column_integrated_dQ2'
# field = 'water_vapor_path'
da1 = prediction[field].isel(tile=itile).mean('time')
da2 = target[field].isel(tile=itile).mean('time')

if res == 'c48':
    vmax = 1
else:
    vmax = 4

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

plt.sca(ax[0])
plt.pcolormesh(lon, lat, box(da1), vmin=-vmax, vmax=vmax, cmap='BrBG')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('prediction')

plt.sca(ax[1])
plt.pcolormesh(lon, lat, box(da2), vmin=-vmax, vmax=vmax, cmap='BrBG')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('target')

plt.xlim((lims[2], lims[3]))
plt.ylim((lims[0], lims[1]))

plt.suptitle(f'{field}', fontsize=20)

plt.savefig(f'map_offline{model_tag}.png', bbox_inches='tight')
plt.close()
