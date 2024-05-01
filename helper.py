import numpy as np
import pandas as pd
import xarray as xr

def unstack_sample(da):
    assert da.dims == ('_fv3net_sample',), 'da must have a single dimension _fv3net_sample'
    dim_stacked = da.dims[0] # should be '_fv3net_sample'

    dims = list(da.coords)
    if dims == ['tile', 'time', 'x', 'y']:
        dims = ['tile', 'time', 'y', 'x']
    else:
        print('Warming: unknown dims', dims)

    multi_index = pd.MultiIndex.from_arrays([da[dim].values for dim in dims], names=dims)
    da_multi_index = xr.DataArray(da.values, coords={dim_stacked: multi_index})
    da_unstacked = da_multi_index.unstack(dim_stacked)

    return da_unstacked

def downsample(da, fac=2):
    return da[..., ::fac, ::fac]

from scipy.linalg import block_diag

def coarse_grain(data4d, factor=1):
    """Coarse grain the last two dimensions of input data by a factor >= 1
    
    Args:
        factor (int)
    """
    
    ndim = len(data4d.dims)
    
    yaxis = data4d[data4d.dims[-2]]
    xaxis = data4d[data4d.dims[-1]]    
    data = data4d.values
    
    # parameters needed for coarse-graining
    xlen = data.shape[-1]
    ylen = data.shape[-2]
    xquotient = xlen//factor
    yquotient = ylen//factor
    ysouth = (ylen - yquotient*factor)//2
    ynorth = ysouth + yquotient*factor
    
    # helper matrices
    onecol = np.ones((factor, 1))/factor # a column vector
    ones = (onecol,)*xquotient
    right = block_diag(*ones)
    onerow = np.ones((1, factor))/factor # a row vector
    ones = (onerow,)*yquotient
    left = block_diag(*ones)
    
    # do the work
    xcoarse = np.dot(xaxis.values, right)
    ycoarse = np.dot(left, yaxis.values[ysouth:ynorth]).flatten()
    
    if ndim == 4:
        taxis = data4d[data4d.dims[0]]
        zaxis = data4d[data4d.dims[1]]
        coarse = np.array([[np.dot( np.dot(left, data[it,iz,ysouth:ynorth,:]), right ) for iz in range(data.shape[1])] for it in range(data.shape[0])])
        da = xr.DataArray(coarse, dims=data4d.dims, coords=[taxis,zaxis,ycoarse,xcoarse], name=data4d.name)
    elif ndim == 3:
        taxis = data4d[data4d.dims[0]]
        coarse = np.array([np.dot( np.dot(left, data[it,ysouth:ynorth,:]), right ) for it in range(data.shape[0])])
        da = xr.DataArray(coarse, dims=data4d.dims, coords=[taxis,ycoarse,xcoarse], name=data4d.name)
    elif ndim == 2:
        coarse = np.dot( np.dot(left, data[ysouth:ynorth,:]), right )
        da = xr.DataArray(coarse, dims=data4d.dims, coords=[ycoarse,xcoarse], name=data4d.name)
    else:
        print("check ndim")
        
    return da

def latlon(da, lat, lon, lims=[]):
    """select a lat-lon box on the native grid"""

    if len(lims) == 0:
        lims = [lat.min(), lat.max(), lon.min(), lon.max()]

    da = da.where(lims[0] < lat).where(lat < lims[1]).where(lims[2] < lon).where(lon < lims[3])

    return da

def change_lon(lon):
    """change lon value; order not changed"""

    to_shift = lon > 180
    lon = lon - 360*to_shift
    
    return lon

def _change_lon_axis_1d(da):
    xname = da.dims[-1]

    return xr.DataArray(da.values, coords=[change_lon(da[xname].values)], dims=da.dims)

def _change_lon_axis_2d(da):
    xname = da.dims[-1]
    yname = da.dims[-2]

    return xr.DataArray(da.values, coords=[da[yname].values, change_lon(da[xname].values)], dims=da.dims)

def change_lon_axis(da):
    """Designed for regional data in the western hemisphere
    From lon = 0-360 to lon = -180-180
    Change the lon axis only, not the data
    """

    if len(da.dims) == 1:
        return _change_lon_axis_1d(da)

    return op_2d_to_nd(_change_lon_axis_2d, da)

def _roll_lon_1d(da):
    """from lon = 0-360 to lon = -180-180"""
    
    xvec = da[da.dims[-1]].values
    nx = len(xvec)
    
    if np.max(xvec) - np.min(xvec) < 350: # do not apply to regional data
        return da
    
    if np.max(xvec) < 180: # no need to roll
        return da
    
    da_out = xr.DataArray(np.roll(da.values, nx//2, axis=-1), 
                       coords=[np.concatenate([xvec[nx//2:] - 360, xvec[:nx//2]])], 
                       dims=[da.dims[-1]])
    
    try: # preserve long_name
        da_out.attrs['long_name'] = da.attrs['long_name']
    except KeyError:
        pass
    
    return da_out

def _roll_lon_2d(da):
    """from lon = 0-360 to lon = -180-180"""
    
    xvec = da[da.dims[-1]].values
    nx = len(xvec)
    
    if np.max(xvec) - np.min(xvec) < 350: # do not apply to regional data
        return da
    
    if np.max(xvec) < 180: # no need to roll
        return da
    
    da_out = xr.DataArray(np.roll(da.values, nx//2, axis=-1), 
                       coords=[da[da.dims[-2]], np.concatenate([xvec[nx//2:] - 360, xvec[:nx//2]])], 
                       dims=[da.dims[-2], da.dims[-1]])
    
    try: # preserve long_name
        da_out.attrs['long_name'] = da.attrs['long_name']
    except KeyError:
        pass
    
    return da_out

def roll_lon(da):
    """from lon = 0-360 to lon = -180-180"""

    if len(da.dims) == 1:
        return _roll_lon_1d(da)

    return op_2d_to_nd(_roll_lon_2d, da)
    
def crop(da, xlim, ylim):
    """Crop the given DataArray to the given boundaries; handels different lon conventions
    """

    xname = da.dims[-1]
    yname = da.dims[-2]

    if xlim[0] < da[xname].values[0] and (da[xname].values[-1] - da[xname].values[0]) > 350: # if out of bounds can be fixed by roll_lon
        if xlim[1] < da[xname].values[0]: # if box is entirely in the western hemisphere
            da = da.sel({xname: slice(xlim[0]+360, xlim[1]+360), yname: slice(ylim[0], ylim[1])})
            da = change_lon_axis(da) # fast
        else: # if box crosses 0 deg
            da = roll_lon(da) # slow
            da = da.sel({xname: slice(xlim[0], xlim[1]), yname: slice(ylim[0], ylim[1])})
    else: # don't need to roll
        da = da.sel({xname: slice(xlim[0], xlim[1]), yname: slice(ylim[0], ylim[1])})

    return da