import numpy as np
import xarray as xr

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


