"""
bash
module load python/3.9
source activate fv3net-prognostic-run
python
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
if snapshot:
    tfinal = ds_ml.time[n_per_day*n_day_analyzed - 1]

    da_base = ds_base[field].isel(tile=itile).sel(time=tfinal)
    da_nudged = ds_nudged[field].isel(tile=itile).sel(time=tfinal)
    da_ml = ds_ml[field].isel(tile=itile).sel(time=tfinal)
else:
    if field == 'total_precipitation_rate':
        itrange = range(n_per_day*0, n_per_day*n_day_analyzed)
        trange_full = ds_ml.time[itrange] # 2020, 1, 29, 3, 0
        trange_half = ds_ml.time[itrange] - datetime.timedelta(minutes=90) # 2020, 1, 29, 1, 30
        trange_shield = trange_full

        da_base = ds_base[field].isel(tile=itile).sel(time=trange_half).mean('time')
        da_nudged = ds_nudged[field].isel(tile=itile).sel(time=trange_half).mean('time')
        da_ml = ds_ml[field].isel(tile=itile).sel(time=trange_full).mean('time')
    else: # if field == 'PRATEsfc' or field == 'snowd':
        trange = ds_ml.time[n_per_day*0:n_per_day*n_day_analyzed]
        if field == 'snowd':
            trange_shield = trange[7::8]
        else:
            trange_shield = trange

        da_base = ds_base[field].isel(tile=itile).sel(time=trange).mean('time')
        da_nudged = ds_nudged[field].isel(tile=itile).sel(time=trange).mean('time')
        da_ml = ds_ml[field].isel(tile=itile).sel(time=trange).mean('time')

if field == 'PRATEsfc' or field == 'total_precipitation_rate':
    da_base = da_base*86400
    da_nudged = da_nudged*86400
    da_ml = da_ml*86400
    da_shield = pr
else:
    da_shield = snowd

if snapshot:
    da_shield = da_shield.sel(time=tfinal)
else:
    da_shield = da_shield.sel(time=trange_shield).mean('time')
da_shield = coarse_grain(da_shield, 8)

lat = ds_land['lat'].isel(tile=itile).isel(time=0).values
lon = ds_land['lon'].isel(tile=itile).isel(time=0).values - 360
land_mask = ds_land['SLMSKsfc'].isel(tile=itile).isel(time=0)

######################## plot maps ########################
fig, ax = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)

plt.sca(ax[0])
plt.pcolormesh(box_shield(da_shield).grid_xt, box_shield(da_shield).grid_yt, box_shield(da_shield), vmin=0, vmax=vmax, cmap='Blues')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('X-SHiELD')

plt.sca(ax[1])
plt.pcolormesh(lon, lat, box(da_nudged), vmin=0, vmax=vmax, cmap='Blues')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('Nudged')

plt.sca(ax[2])
plt.pcolormesh(lon, lat, box(da_base), vmin=0, vmax=vmax, cmap='Blues')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('Baseline')

plt.sca(ax[3])
plt.pcolormesh(lon, lat, box(da_ml), vmin=0, vmax=vmax, cmap='Blues')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('ML-corrected')

plt.xlim((lims[2], lims[3]))
plt.ylim((lims[0], lims[1]))

plt.suptitle(f'{field} {units} over {n_day_analyzed} days', fontsize=20)

plt.savefig(f'map_{field}.png', bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

plt.sca(ax[0])
plt.pcolormesh(lon, lat, box(da_base - da_nudged), vmin=-vmax/2, vmax=vmax/2, cmap='BrBG')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('Baseline $-$ Nudged')

plt.sca(ax[1])
plt.pcolormesh(lon, lat, box(da_ml - da_nudged), vmin=-vmax/2, vmax=vmax/2, cmap='BrBG')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('ML-corrected $-$ Nudged')

plt.xlim((lims[2], lims[3]))
plt.ylim((lims[0], lims[1]))

plt.suptitle(f'$\Delta${field} {units}', fontsize=20)

plt.savefig(f'diff_{field}.png', bbox_inches='tight')
plt.close()

######################## time series ########################
da_base = ds_base[field].isel(tile=itile)
da_nudged = ds_nudged[field].isel(tile=itile)
da_ml = ds_ml[field].isel(tile=itile)

if field == 'PRATEsfc' or field == 'total_precipitation_rate':
    da_base = da_base*86400
    da_nudged = da_nudged*86400
    da_ml = da_ml*86400

trange_all = ds_land.time
if field == 'PRATEsfc' or field == 'total_precipitation_rate':
    da_shield = pr.sel(time=trange_all)
else:
    da_shield = snowd.sel(time=trange_all[7::8])
da_shield = coarse_grain(da_shield, 8)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

if field == 'total_precipitation_rate':
    xydims = ['y', 'x']
else:
    xydims = ['grid_yt', 'grid_xt']

## assumption: running length X-SHiELD >= nudged >= baseline, ML-corrected
if field == 'PRATEsfc' or field == 'total_precipitation_rate':
    thresh = 0

    xvec_full = range(0, len(trange_all))

    plt.plot(xvec_full, box_shield(da_shield.where(da_shield > thresh)).mean(['grid_yt', 'grid_xt']), label='X-SHiELD', c='k', ls='--', marker='x')

    plt.plot(xvec_full, box(da_nudged.where(da_nudged > thresh)).mean(xydims), label='Nudged', c='k', lw=2)

    init = abs(trange_all - da_base.time[0]).argmin().values # find the index of the first time step in the baseline run
    xvec_base = range(init, init+len(da_base.time))
    plt.plot(xvec_base, box(da_base.where(da_base > thresh)).mean(xydims), label='Baseline', c='tab:blue', lw=2)

    init = abs(trange_all - da_ml.time[0]).argmin().values # find the index of the first time step in the ML-corrected run
    xvec_ml = range(init, init+len(da_ml.time))
    plt.plot(xvec_ml, box(da_ml.where(da_ml > thresh)).mean(xydims), label='ML-corrected', c='tab:orange', lw=2)

    plt.xticks(xvec_full[7::16], [trange_all.values[i].strftime('%m-%d') for i in xvec_full[7::16]], rotation=45) # label every 16 time steps
    plt.title(f'Domain mean {field}')
else:
    xvec_full = range(0, len(trange_all))

    plt.plot(xvec_full[7::8], box_shield(da_shield).max(xydims), label='X-SHiELD', c='k', ls='--', marker='x')

    plt.plot(xvec_full, box(da_nudged).max(xydims), label='Nudged', c='k', lw=2)

    init = abs(trange_all - da_base.time[0]).argmin().values
    xvec_base = range(init, init+len(da_base.time))
    plt.plot(xvec_base, box(da_base).max(xydims), label='Baseline', c='tab:blue', lw=2)

    init = abs(trange_all - da_ml.time[0]).argmin().values
    xvec_ml = range(init, init+len(da_ml.time))
    plt.plot(xvec_ml, box(da_ml).max(xydims), label='ML-corrected', c='tab:orange', lw=2)

    plt.xticks(xvec_full[7::16], [trange_all.values[i].strftime('%m-%d') for i in xvec_full[7::16]], rotation=45)
    plt.title(f'Domain max {field}')

plt.legend(loc='best')
plt.ylabel(units)
ylim = plt.gca().get_ylim(); plt.ylim((0, ylim[-1]))
plt.savefig(f'ts_{field}.png', bbox_inches='tight')
plt.close()
