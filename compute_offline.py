"""
source activate fv3net-image
"""

##### from workflows/diagnostics/fv3net/diagnostics/offline/compute.py
import argparse
import json
# import logging
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

class args: # this is a hack to replace the parser
    evaluation_grid = 'c48'
    data_yaml = '/home/hsiehtl/nudge-to-fine-stellar-workflow/configs/machine-learning/tq-testing-data.yaml'
    catalog_path = '/home/hsiehtl/nudge-to-fine-stellar-workflow/configs/catalog.yaml'
    model_path = '/scratch/cimes/hsiehtl/C48/models/tq-model'
    output_path = '/scratch/cimes/hsiehtl/C48/offline/test/tq-model' # to be updated
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

prediction.to_netcdf("prediction.nc")
target.to_netcdf("target.nc")
########################################

##### plot maps
import matplotlib.pyplot as plt
from helper import coarse_grain, latlon, crop

def box(da):
    return latlon(da, lat, lon, lims)

ds_land = xr.open_zarr('/scratch/cimes/hsiehtl/C48/simulations/nudged/sfc_dt_atmos.zarr') # for land_mask

# itile = 2; lims = [42, 52, -127, -117] # PNW # to compare the ratio between two mountains
itile = 2; lims = [42, 62, -135, -115] # PNW+
# itile = 4; lims = [34, 42, -124, -117] # CA-NV # to compare the first 5 days
# itile = 4; lims = [12, 42, -134, -104] # CA-NV+

lat = ds_land['lat'].isel(tile=itile).values
lon = ds_land['lon'].isel(tile=itile).values - 360
land_mask = ds_land['SLMSKsfc'].isel(tile=itile).isel(time=0)

da1 = prediction['column_integrated_Q2'].isel(tile=itile).mean('time')
da2 = target['column_integrated_Q2'].isel(tile=itile).mean('time')

vmax = 1

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

plt.sca(ax[0])
plt.pcolormesh(lon, lat, box(da1), vmin=0, vmax=vmax, cmap='Blues')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('prediction')

plt.sca(ax[1])
plt.pcolormesh(lon, lat, box(da2), vmin=0, vmax=vmax, cmap='Blues')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('target')

plt.xlim((lims[2], lims[3]))
plt.ylim((lims[0], lims[1]))

# plt.suptitle(f'{field} {units} over {n_day_analyzed} days', fontsize=20)

plt.savefig(f'map_test.png', bbox_inches='tight')
plt.close()
########################################

# # save model senstivity figures- these exclude derived variables
# fig_input_sensitivity = plot_input_sensitivity(model, ds_predicted, horizontal_dims)
# if fig_input_sensitivity is not None:
#     with fsspec.open(
#         os.path.join(
#             args.output_path, "model_sensitivity_figures", INPUT_SENSITIVITY
#         ),
#         "wb",
#     ) as f:
#         fig_input_sensitivity.savefig(f)

# mapper = _get_data_mapper_if_exists(config)
# if mapper is not None:
#     snapshot_timestamp = (
#         args.snapshot_time
#         or sorted(getattr(config, "timesteps", list(mapper.keys())))[0]
#     )
#     snapshot_time = vcm.parse_datetime_from_str(snapshot_timestamp)

#     ds_snapshot = select_snapshot(ds_predicted, snapshot_time)

#     vertical_vars = [
#         var for var in model.output_variables if is_3d(ds_snapshot[var])
#     ]
#     ds_snapshot = insert_column_integrated_vars(ds_snapshot, vertical_vars)
#     predicted_vars = [
#         var for var in ds_snapshot if "derivation" in ds_snapshot[var].dims
#     ]

#     # add snapshotted prediction to saved diags.nc
#     ds_diagnostics = ds_diagnostics.merge(
#         safe.get_variables(ds_snapshot, predicted_vars).rename(
#             dict(
#                 **{v: f"{v}_snapshot" for v in predicted_vars}, time="time_snapshot"
#             )
#         )
#     )

#     ds_transect = _get_transect(
#         ds_snapshot, evaluation_grid, vertical_vars, config.ptop,
#     )
#     _write_nc(ds_transect, args.output_path, TRANSECT_NC_NAME)

# ds_diagnostics = _add_derived_diagnostics(ds_diagnostics)

# _write_nc(
#     ds_diagnostics, args.output_path, DIAGS_NC_NAME,
# )

# # convert and output metrics json
# metrics = {
#     var: ds_scalar_metrics[var].item() for var in ds_scalar_metrics.data_vars
# }
# with fsspec.open(os.path.join(args.output_path, METRICS_JSON_NAME), "w") as f:
#     json.dump(metrics, f, indent=4)

# metadata = {}
# metadata["model_path"] = args.model_path
# metadata["data_config"] = dataclasses.asdict(config)
# with fsspec.open(os.path.join(args.output_path, METADATA_JSON_NAME), "w") as f:
#     json.dump(metadata, f, indent=4)

# # logger.info(f"Finished processing dataset diagnostics and metrics.")