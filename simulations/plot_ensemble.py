"""
bash
module load python/3.9
source activate fv3net-image
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import datetime
plt.rcParams.update({'font.size': 16})
## add path
import sys
sys.path.append('/ncrc/home2/Tsung-Lin.Hsieh/fv3net_helper')
from helper import coarse_grain, latlon, crop

def box(da):
    return latlon(da, lat, lon, lims)

def box_shield(da):
    cropped = crop(da, (lims[2], lims[3]), (lims[0], lims[1]))

    if len(cropped.values.flatten()) == 0:
        return xr.full_like(da, fill_value=np.nan)

    return cropped

def get_pdf(da):
    da1d = da.values.flatten()
    da1d = da1d[np.isfinite(da1d)] # remove nan

    count, edges = np.histogram(da1d, bins=40, range=(0, 400), density=False)
    count = xr.DataArray(count, coords=[(edges[:-1] + edges[1:])/2], dims=['rain'], name='pdf')

    dx = edges[1] - edges[0]
    pdf = count/len(da1d)/dx

    return pdf

def get_weighted_hist(hgt, da):
    hgt1d = hgt.values.flatten()
    hgt1d = hgt1d[np.isfinite(hgt1d)] # remove nan

    da1d = da.values.flatten()
    da1d = da1d[np.isfinite(da1d)] # remove nan

    count, edges = np.histogram(hgt1d, bins=30, range=(0, 3000), density=False, weights=da1d)
    count = xr.DataArray(count, coords=[(edges[:-1] + edges[1:])/2], dims=['sfc_hgt'], name='hist')

    dx = edges[1] - edges[0]
    pdf = count/dx

    return pdf
    
## constants
ds_land = xr.open_zarr('/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384/simulations/nudged/sfc_dt_atmos.zarr')
pr = xr.open_dataarray('/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/pr_X-SHiELD_2020Jan19-2020Mar13_CA-NV.nc')
snowd = xr.open_dataarray('/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/snowd_X-SHiELD_2020Jan19-2020Mar13_CA-NV.nc')

## specify options
# itile = 2; lims = [42, 52, -127, -117] # PNW # to compare the ratio between two mountains
# itile = 2; lims = [42, 62, -135, -115] # PNW+
itile = 4; lims = [34, 42, -124, -117] # CA-NV # to compare the first 5 days
# itile = 4; lims = [12, 42, -134, -104] # CA-NV+

exprs_base = {
    'baseline_0129_40day': 40, 
    'baseline_01290300_6day': 6, 
    'baseline_01290600_6day': 6, 
    'baseline_01290900_6day': 6, 
    'baseline_01291200_6day': 6,
    # 'baseline_0225_6day': 6, 
    # 'baseline_0303_6day': 6, 
    }
    
# exprs_ml = {
#     'ml-corrected_0129_6day_wna': 6,
#     'ml-corrected_0129_6day_wna_seed1': 6,
#     'ml-corrected_0129_6day_wna_seed2': 6,
#     'ml-corrected_0129_6day_wna_seed3': 6,
#     'ml-corrected_01291200_6day_wna_seed1': 6,
#     'ml-corrected_01291200_6day_wna_seed2': 6,
#     'ml-corrected_01291200_6day_wna_seed3': 10,
#     'ml-corrected_0130_6day_wna': 6,
#     'ml-corrected_0131_6day_wna': 6,
#     }
    
# exprs_ml = {
#     'ml-corrected_0129_6day_sfc1000': 6,
#     'ml-corrected_01290300_6day_sfc1000': 6,
#     'ml-corrected_01290600_6day_sfc1000': 6,
#     'ml-corrected_01290900_6day_sfc1000': 10,
#     }
    
exprs_ml = {
    'ml-corrected_0129_6day_sfc1000': 6,
    'ml-corrected_0129_sfc1000_no0202-04_seed1': 6,
    'ml-corrected_0129_sfc1000_no0202-04_seed2': 6,
    'ml-corrected_0129_sfc1000_no0202-04_seed3': 6,
    }

field = 'total_precipitation_rate'; units = '[mm/day]'; zarrname = 'diags'
# field = 'snowd'; units = '[m]'; zarrname = 'sfc_dt_atmos'

## load files
ds_nudged = xr.open_zarr(f'/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384/simulations/nudged/{zarrname}.zarr')
dss_base = [xr.open_zarr(f'/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384/simulations/{expr_base}/{zarrname}.zarr') for expr_base in exprs_base]
dss_ml = [xr.open_zarr(f'/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384/simulations/{expr_ml}/{zarrname}.zarr') for expr_ml in exprs_ml]

da_nudged = ds_nudged[field].isel(tile=itile)
das_base = [ds_base[field].isel(tile=itile) for ds_base in dss_base]
das_ml = [ds_ml[field].isel(tile=itile) for ds_ml in dss_ml]

if field == 'PRATEsfc' or field == 'total_precipitation_rate':
    da_nudged = da_nudged*86400
    das_base = [da_base*86400 for da_base in das_base]
    das_ml = [da_ml*86400 for da_ml in das_ml]

if 'time' in ds_land['lat'].dims:
    lat = ds_land['lat'].isel(tile=itile).isel(time=0).values
    lon = ds_land['lon'].isel(tile=itile).isel(time=0).values - 360
else:
    lat = ds_land['lat'].isel(tile=itile).values
    lon = ds_land['lon'].isel(tile=itile).values - 360
land_mask = ds_land['SLMSKsfc'].isel(tile=itile).isel(time=0)

########## plot time series ##########
trange_all = ds_nudged.time

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

if field == 'total_precipitation_rate':
    xydims = ['y', 'x']
else:
    xydims = ['grid_yt', 'grid_xt']

## assumption: running length X-SHiELD >= nudged >= baseline, ML-corrected
if field == 'PRATEsfc' or field == 'total_precipitation_rate':
    thresh = 0

    xvec_full = range(0, len(trange_all))

    plt.plot(xvec_full, box(da_nudged.where(da_nudged > thresh)).mean(xydims), label='Nudged', c='k', lw=2)

    for da_base in das_base:
        init = abs(trange_all - da_base.time[0]).argmin().values
        xvec_base = range(init, init+len(da_base.time))
        plt.plot(xvec_base, box(da_base.where(da_base > thresh)).mean(xydims), label='Baseline', c='tab:blue', lw=1)
        plt.plot(xvec_base[0], box(da_base.where(da_base > thresh)).mean(xydims)[0], 'o', c='tab:blue', markersize=10)

    for da_ml in das_ml:
        init = abs(trange_all - da_ml.time[0]).argmin().values # find the index of the first time step in the ML-corrected run
        xvec_ml = range(init, init+len(da_ml.time))
        plt.plot(xvec_ml, box(da_ml.where(da_ml > thresh)).mean(xydims), label='ML-corrected', c='tab:orange', lw=1)
        plt.plot(xvec_ml[0], box(da_ml.where(da_ml > thresh)).mean(xydims)[0], 'x', c='tab:orange', markersize=10)

    plt.xticks(xvec_full[::8], [trange_all.values[i].strftime('%m-%d') for i in xvec_full[::8]], rotation=45) # label every 8 time steps
    plt.title(f'Domain mean {field}')
else:
    xvec_full = range(0, len(trange_all))

    plt.plot(xvec_full, box(da_nudged).max(xydims), label='Nudged', c='k', lw=2)

    for da_base in das_base:
        init = abs(trange_all - da_base.time[0]).argmin().values
        xvec_base = range(init, init+len(da_base.time))
        plt.plot(xvec_base, box(da_base).max(xydims), label='Baseline', c='tab:blue', lw=1)
        plt.plot(xvec_base[0], box(da_base).max(xydims)[0], 'o', c='tab:blue', markersize=10)

    for da_ml in das_ml:
        init = abs(trange_all - da_ml.time[0]).argmin().values
        xvec_ml = range(init, init+len(da_ml.time))
        plt.plot(xvec_ml, box(da_ml).max(xydims), label='ML-corrected', c='tab:orange', lw=1)
        plt.plot(xvec_ml[0], box(da_ml).max(xydims)[0], 'x', c='tab:orange', markersize=10)

    plt.xticks(xvec_full[::8], [trange_all.values[i].strftime('%m-%d') for i in xvec_full[::8]], rotation=45)
    plt.title(f'Domain max {field}')

## make legend unique
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best')

plt.xlim((0, 7*8))
plt.ylabel(units)
ylim = plt.gca().get_ylim(); plt.ylim((0, ylim[-1]))
plt.savefig(f'ts_ens_{field}.png', bbox_inches='tight')
plt.close()

########## plot map ##########
time_common = ds_nudged['time'][2*8:6*8] # day 2 to day 6 from 0129

da_nudged = ds_nudged[field].isel(tile=itile).sel(time=time_common)
das_base = [ds_base[field].isel(tile=itile).sel(time=time_common) for ds_base in dss_base]
das_ml = [ds_ml[field].isel(tile=itile).sel(time=time_common) for ds_ml in dss_ml]

if field == 'PRATEsfc' or field == 'total_precipitation_rate':
    da_nudged = da_nudged.mean('time')*86400
    das_base = [da_base.mean('time')*86400 for da_base in das_base]
    das_ml = [da_ml.mean('time')*86400 for da_ml in das_ml]
    da_shield = pr.sel(time=(time_common - datetime.timedelta(minutes=90))).mean('time')
    vmax = 32
else:
    da_nudged = da_nudged.isel(time=-1)
    das_base = [da_base.isel(time=-1) for da_base in das_base]
    das_ml = [da_ml.isel(time=-1) for da_ml in das_ml]
    da_shield = snowd.sel(time=time_common[-1])
    vmax = 2
# da_shield = coarse_grain(da_shield, 8)

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
plt.pcolormesh(lon, lat, xr.concat([box(da_base) for da_base in das_base], dim='ens').mean('ens'), vmin=0, vmax=vmax, cmap='Blues')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('Baseline')

plt.sca(ax[3])
plt.pcolormesh(lon, lat, xr.concat([box(da_ml) for da_ml in das_ml], dim='ens').mean('ens'), vmin=0, vmax=vmax, cmap='Blues')
plt.colorbar()
plt.contour(lon, lat, box(land_mask), levels=[0.5], colors='k')
plt.title('ML-corrected')

plt.xlim((lims[2], lims[3]))
plt.ylim((lims[0], lims[1]))

if field == 'PRATEsfc' or field == 'total_precipitation_rate':
    plt.suptitle(f"ensemble mean {field} {units}, {time_common.values[0].strftime('%m-%d')} to {time_common.values[-1].strftime('%m-%d')}", fontsize=20)
else:
    plt.suptitle(f"ensemble mean {field} {units}, {time_common.values[-1].strftime('%m-%d')}", fontsize=20)

plt.savefig(f'map_ens_{field}.png', bbox_inches='tight')
plt.close()

if field == 'snowd':
    exit()

########## plot pdf ##########
time_common = ds_nudged['time'][2*8:6*8] # day 2 to day 6 from 0129

da_nudged = ds_nudged[field].isel(tile=itile).sel(time=time_common)
das_base = [ds_base[field].isel(tile=itile).sel(time=time_common) for ds_base in dss_base]
das_ml = [ds_ml[field].isel(tile=itile).sel(time=time_common) for ds_ml in dss_ml]

if field == 'PRATEsfc' or field == 'total_precipitation_rate':
    da_nudged = da_nudged*86400
    das_base = [da_base*86400 for da_base in das_base]
    das_ml = [da_ml*86400 for da_ml in das_ml]

fig, ax = plt.subplots(1, 1, figsize=(6, 5), facecolor='w')

pdf_nudged = get_pdf(box(da_nudged))
pdf_nudged.plot(label='Nudged', c='k', lw=2)

for da_base in das_base:
    pdf_base = get_pdf(box(da_base))
    pdf_base.plot(label='Baseline', c='tab:blue', lw=1)

for da_ml in das_ml:
    pdf_ml = get_pdf(box(da_ml))
    pdf_ml.plot(label='ML-corrected', c='tab:orange', lw=1)

plt.xlabel('Precipitation rate [mm/day]')
plt.ylabel('PDF [(mm/day)$^{-1}$]')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.xlim((0, 200))
plt.ylim((5e-6, 1e-1))
plt.yscale('log')
plt.title('3-hrly mean precipitation rate', fontsize=18)

plt.savefig(f'pdf_ens_{field}.png', bbox_inches='tight')
plt.close()

########## plot histogram by surface height ##########
hgt = xr.open_zarr(f'/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384/simulations/nudged/state_after_timestep.zarr')['surface_geopotential'].isel(time=0).isel(tile=itile)/9.81

box_hgt = box(hgt)
box_hgt = box_hgt.expand_dims(time=time_common)

fig, ax = plt.subplots(1, 1, figsize=(6, 5), facecolor='w')

pdf_nudged = get_weighted_hist(box_hgt, box(da_nudged))
(pdf_nudged/len(time_common)).plot(label='Nudged', c='k', lw=2)

for da_base in das_base:
    pdf_base = get_weighted_hist(box_hgt, box(da_base))
    (pdf_base/len(time_common)).plot(label='Baseline', c='tab:blue', lw=1)

for da_ml in das_ml:
    pdf_ml = get_weighted_hist(box_hgt, box(da_ml))
    (pdf_ml/len(time_common)).plot(label='ML-corrected', c='tab:orange', lw=1)

# pdf_nudged = get_weighted_hist(box_hgt, box(da_nudged))
# pdfs_base = xr.concat([get_weighted_hist(box_hgt, box(da_base)) for da_base in das_base], dim='ens')
# pdfs_ml = xr.concat([get_weighted_hist(box_hgt, box(da_ml)) for da_ml in das_ml], dim='ens')
# (pdf_nudged/len(time_common)).plot(label='Nudged', c='k', lw=2)
# (pdfs_base.mean('ens')/len(time_common)).plot(label='Baseline', c='tab:blue', lw=2)
# (pdfs_ml.mean('ens')/len(time_common)).plot(label='ML-corrected', c='tab:orange', lw=2)

plt.xlabel('Surface height [m]')
plt.ylabel('Accumulated precip per bin [mm/day/m]')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.xlim((0, 3000))
plt.ylim((0, 10))
plt.title('3-hrly mean precipitation rate', fontsize=18)

plt.savefig(f'pdf_ens_{field}_hgt.png', bbox_inches='tight')
plt.close()
