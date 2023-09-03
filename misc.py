
import numpy as np
import scipy
from scipy import signal
from astropy import convolution


def array2bins(x, dx, crpix, n=None, offset=2):
    dx = np.abs(dx)
    if n is None:
        nhalf = int((crpix-(np.nanmin(x)))/dx + 0.1)
        n = nhalf*2 + offset*2
    else:
        n = np.abs(n) + offset*2
    nhalf = int((crpix-(np.nanmin(x)-offset*dx))/dx + 0.1)
    ret = np.zeros(n)
    ret[nhalf] = crpix
    ret[:nhalf] = crpix - dx*np.arange(nhalf, 0, -1)
    ret[nhalf:] = crpix + dx*np.arange(0, int(n-nhalf), 1)
    return ret, nhalf+1 # header indexing begins with 1 (NOT 0)

def filter_gaussian(data,pixsize,x_stddev,y_stddev=None,theta=0.0,mode='same',method='direct',**kwargs):
    x_stddev_pix = x_stddev/pixsize
    y_stddev_pix = None if y_stddev is None else y_stddev/pixsize
    kernel = astropy.convolution.Gaussian2DKernel(x_stddev_pix,y_stddev_pix,theta)
    if len(np.shape(data)) == 3:
        x = data[0]
    else:
        x = data
    ret = scipy.signal.convolve(x, kernel, mode=mode, method=method, **kwargs)

    if len(np.shape(data)) == 3:
        return np.reshape(ret,(1,*np.shape(ret)))
    return ret

# functions for mask

def mask(x,mask):
    '''
    return masked x array with mask array
    (masked element will be replaced with np.nan)
    mask: 1 --> masked, 0 --> not masked
    '''
    ret = np.copy(x)
    ret[mask>0.001] = np.nan
    return ret


# aperture photometry

def calc_integral(cx,cy,x,y,v,r,noise=0,pixarea=None):
    '''
    return ingral of `v` from the center (`cx`,`cy`) [deg] to radius `r` [arcsec]
    target map info: `x` and `y` [deg] for coordinate, `v` [*/arcsec^2] to be calculated
    '''
    dist = np.sqrt((x-cx)**2 + (y-cy)**2)*3600 # [arcsec]

    if not hasattr(r, "__iter__"):
        rlist = [r]
    else:
        rlist = r

    vf = v.flatten()
    distf = dist.flatten()
    if pixarea is None:
        ret = np.array([(np.nanmean(vf[distf<ir])-noise) * np.pi*ir**2 for ir in rlist])
    else:
        ret = np.array([(np.nansum(vf[distf<ir] - noise)) * pixarea  for ir in rlist])

    if not hasattr(r, "__iter__"):
        return ret[0]
    return ret

def calc_noise(cx,cy,x,y,v,r0,r1,ret_rms=False):
    dist = np.sqrt((x-cx)**2 + (y-cy)**2)*3600 # [arcsec]

    vf = v.flatten()
    distf = dist.flatten()
    ss = (distf>=r0) & (distf<r1)
    noise = np.nanmean(vf[ss])
    noiserms = np.nanstd(vf[ss])

    if ret_rms:
        return noise, noiserms
    else:
        return noise

