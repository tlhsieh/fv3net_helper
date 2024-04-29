"""
bash
module load python/3.9
source activate fv3net-prognostic-run
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime
plt.rcParams.update({'font.size': 16})
from helper import coarse_grain, latlon, crop

def box(da):
    return latlon(da, lat, lon, lims)

def box_shield(da):
    cropped = crop(da, (lims[2], lims[3]), (lims[0], lims[1]))

    if len(cropped.values.flatten()) == 0:
        return xr.full_like(da, fill_value=np.nan)

    return cropped

## constants
n_per_day = 8
ds_land = xr.open_zarr('/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384/simulations/nudged/sfc_dt_atmos.zarr') # for land_mask
pr = xr.open_dataarray('/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/pr_X-SHiELD_2020Jan19-2020Mar13_CA-NV.nc')
snowd = xr.open_dataarray('/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/snowd_X-SHiELD_2020Jan19-2020Mar13_CA-NV.nc')

## specify options
# itile = 2; lims = [42, 52, -127, -117] # PNW # to compare the ratio between two mountains
# itile = 2; lims = [42, 62, -135, -115] # PNW+
itile = 4; lims = [34, 42, -124, -117] # CA-NV # to compare the first 5 days
# itile = 4; lims = [12, 42, -134, -104] # CA-NV+

expr_base = 'baseline_0129_40day'
expr_ml = 'ml-corrected_0129_40day'; n_day_analyzed = 40 # number of days to calculate the mean over
# expr_base = 'baseline_0225_6day'
# expr_ml = 'ml-corrected_0225_6day'; n_day_analyzed = 6 # number of days to calculate the mean over
# expr_base = 'baseline_0303_6day'
# expr_ml = 'ml-corrected_0303_6day'; n_day_analyzed = 6 # number of days to calculate the mean over

field = 'total_precipitation_rate'; vmax = 12; units = '[mm/day]'; zarrname = 'diags'; snapshot = False
# field = 'snowd'; vmax = 1; units = '[m]'; zarrname = 'sfc_dt_atmos'; snapshot = True
# # field = 'PRATEsfc'; vmax = 8; units = '[mm/day]'; zarrname = 'sfc_dt_atmos'; snapshot = False

## load files
ds_nudged = xr.open_zarr(f'/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384/simulations/nudged/{zarrname}.zarr')
ds_base = xr.open_zarr(f'/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384/simulations/{expr_base}/{zarrname}.zarr')
ds_ml = xr.open_zarr(f'/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384/simulations/{expr_ml}/{zarrname}.zarr')

## select data
itrange = range(n_per_day*0, n_per_day*n_day_analyzed)
trange_full = ds_ml.time[itrange] # 2020, 1, 29, 3, 0
trange_half = ds_ml.time[itrange] - datetime.timedelta(minutes=90) # 2020, 1, 29, 1, 30
trange_shield = trange_full

da_base = ds_base[field].isel(tile=itile).sel(time=trange_half)
da_nudged = ds_nudged[field].isel(tile=itile).sel(time=trange_half)
da_ml = ds_ml[field].isel(tile=itile).sel(time=trange_full)

da_base = da_base*86400
da_nudged = da_nudged*86400
da_ml = da_ml*86400

############## new code from here ##############
n_time = 8 # since nudged is instantaneous, we need to average it
da_base = da_base.rolling(time=n_time).mean()[n_time-1::n_time]
da_nudged = da_nudged.rolling(time=n_time).mean()[n_time-1::n_time]
da_ml = da_ml.rolling(time=n_time).mean()[n_time-1::n_time]
# da_base = da_base.resample(time='1D').mean()
# da_nudged = da_nudged.resample(time='1D').mean()
# da_ml = da_ml.resample(time='1D').mean()

def get_pdf(da):
    da1d = da.values.flatten()
    da1d = da1d[np.isfinite(da1d)] # remove nan
    count, edges = np.histogram(da1d, bins=201, range=(-1, 401), density=False)
    count = xr.DataArray(count, coords=[(edges[:-1] + edges[1:])/2], dims=['rain'], name='pdf')
    dx = edges[1] - edges[0]
    pdf = count/len(da1d)/dx

    return pdf

lat = ds_land['lat'].isel(tile=itile).isel(time=0).values
lon = ds_land['lon'].isel(tile=itile).isel(time=0).values - 360

pdf_base = get_pdf(box(da_base))
pdf_nudged = get_pdf(box(da_nudged))
pdf_ml = get_pdf(box(da_ml))

fig, ax = plt.subplots(1, 1, figsize=(6, 5), facecolor='w')

pdf_nudged.plot(label='Nudged', c='k', lw=2)
pdf_base.plot(label='Baseline', c='tab:blue', lw=2)
pdf_ml.plot(label='ML-corrected', c='tab:orange', lw=2)

plt.xlabel('Precipitation rate [mm/day]')
plt.ylabel('PDF [(mm/day)$^{-1}$]')
plt.legend(loc='upper right')

plt.xlim((0, 150))
plt.ylim((1e-6, 5e-1))
plt.yscale('log')
plt.title('Daily mean precipitation rate', fontsize=18)

plt.savefig(f'pdf_{field}.png', bbox_inches='tight')
plt.close()