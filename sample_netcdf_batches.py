"""
bash
module load python/3.9
source activate fv3net-image
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import os

# def get_subsample_indices(ds_source, ds_target):
#     return get_western_us(ds_source, ds_target)
#     # return get_moist_columns(ds_source, ds_target, make_plot=False)

def get_western_us(ds_source, ds_target):
    ## WA-OR, tile2[334:384, 240:290]
    ctile = (ds_source['tile'] == 2)
    cy = (ds_source['y'] >= 334) & (ds_source['y'] < 384) # lat, reversed
    cx = (ds_source['x'] >= 240) & (ds_source['x'] < 290)

    ## CA-NV, tile4[98:127, 5:50] # CA-NV
    # ctile2 = (ds_source['tile'] == 4)
    # cy2 = (ds_source['y'] >= 98) & (ds_source['y'] < 127)
    # cx2 = (ds_source['x'] >= 5) & (ds_source['x'] < 50) # lat

    ## California + Mexico, tile4[60:180, 0:120]
    ctile3 = (ds_source['tile'] == 4)
    cy3 = (ds_source['y'] >= 60) & (ds_source['y'] < 180)
    cx3 = (ds_source['x'] >= 0) & (ds_source['x'] < 120) # lat

    indices = np.where(ctile & cx & cy | ctile3 & cx3 & cy3)[0]

    if len(indices) < len(ds_target['_fv3net_sample']):
        print('Warning: not enough samples in the western US')

    return indices

def get_moist_columns(ds_source, ds_target, make_plot=False):
    iz = -25 # iz = -25 == p = 700 hPa
    da = ds_source['specific_humidity'].isel(z=iz)

    dv = 1.e-3
    nbins = 10
    bin_edges = np.arange(0, dv*(nbins+1), dv)
    bin_edges[-1] = np.max(da)

    ntarget = len(ds_target['_fv3net_sample'])
    assert np.mod(ntarget, nbins) == 0, 'nbins must divide the output sample size'

    if make_plot: # making sure the bins cover the entire range
        da.plot.hist(bins=bin_edges, edgecolor='k', linewidth=1.5)
        plt.savefig('specific_humidity_histogram.png', bbox_inches='tight')
        plt.close()

    ## collect indices for each bin
    indices = [] # indices to select for each bin
    for i in range(len(bin_edges) - 1):
        vmin = bin_edges[i]
        vmax = bin_edges[i+1]

        ## indices of the source data that fall in the range [vmin, vmax)
        idx = np.where((da >= vmin) & (da < vmax))[0]
        ## randomly select samples
        if len(idx) > ntarget/nbins:
            idx_bin = np.random.choice(idx, int(ntarget/nbins), replace=False)
        else:
            idx_bin = np.random.choice(idx, int(ntarget/nbins), replace=True)
        indices.append(idx_bin)

    indices = np.concatenate(indices)
    indices = np.sort(indices)

    assert len(indices) == ntarget, 'total number of samples must match'

    return indices

if __name__ == '__main__':
    expr_root = '/gpfs/f5/gfdl_w/scratch/Tsung-Lin.Hsieh/C384'

    tag_source = 'noSubsampling_backup'
    tag_target = 'subsampleRatio0p015625'
    # tag_output = 'subsampleRatio0p015625_balanced'
    tag_output = 'subsampleRatio0p015625_wus'
    # tag_target = 'subsampleRatio0p0009765625'
    # tag_output = 'subsampleRatio0p0009765625_balanced'

    ## make new directory
    if not os.path.exists(f'{expr_root}/ml-data_{tag_output}'):
        os.system(f'mkdir -p {expr_root}/ml-data_{tag_output}')

    ## copy radiative-fluxes.zarr from the source (which is the same as the target)
    if not os.path.exists(f'{expr_root}/ml-data_{tag_output}/radiative-fluxes.zarr'):
        os.system(f'ln -s {expr_root}/ml-data_{tag_target}/radiative-fluxes.zarr {expr_root}/ml-data_{tag_output}/radiative-fluxes.zarr') # link destination can be either tag_source and tag_target

    for dataset in ['training', 'validation']:
        ## get the number of batches
        nbatch = len(os.listdir(f'{expr_root}/ml-data_{tag_source}/{dataset}/tq'))

        ## collect indices for each batch (slow)
        indices = []
        for ibatch in range(nbatch):
            print(f'Processing {dataset}/tq/000{ibatch:02d}.nc')
            ds_source = xr.open_dataset(f'{expr_root}/ml-data_{tag_source}/{dataset}/tq/000{ibatch:02d}.nc')
            ds_target = xr.open_dataset(f'{expr_root}/ml-data_{tag_target}/{dataset}/tq/000{ibatch:02d}.nc')

            ## get subsampling indices
            if tag_output.endswith('balanced'):
                indices.append(get_moist_columns(ds_source, ds_target))
            elif tag_output.endswith('wus'):
                indices.append(get_western_us(ds_source, ds_target))
            else:
                raise ValueError('Unknown tag_output')

        print('Subsamping by a factor of', len(ds_source['_fv3net_sample'])/len(ds_target['_fv3net_sample']))

        for variable in ['fluxes', 'tq', 'uv']:
            ## if directory doesn't exist, make new directory parallel to the source
            if not os.path.exists(f'{expr_root}/ml-data_{tag_output}/{dataset}/{variable}'):
                os.system(f'mkdir -p {expr_root}/ml-data_{tag_output}/{dataset}/{variable}')

            for ibatch in range(nbatch):
                print(f'Processing {dataset}/{variable}/000{ibatch:02d}.nc')
                ds_source = xr.open_dataset(f'{expr_root}/ml-data_{tag_source}/{dataset}/{variable}/000{ibatch:02d}.nc')

                ## save the subsampled data
                ds_source.isel(_fv3net_sample=indices[ibatch]).to_netcdf(f'{expr_root}/ml-data_{tag_output}/{dataset}/{variable}/000{ibatch:02d}.nc')

