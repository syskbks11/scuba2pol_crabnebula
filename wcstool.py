import numpy as np

import astropy
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import reproject

import misc

# functions with wcs

def wcs_reproject(v0,wcs0,wcs1,return_footprint=False):
    '''
    return modified v0 (for wcs0) fitted to wcs1
    '''
    if len(np.shape(v0)) == 3: # cube
        vmap0 = v0[0]
    else:
        vmap0 = v0
    vmap1,nmap1 = reproject.reproject_adaptive(input_data=(vmap0,wcs0.celestial),
                                               output_projection=wcs1.celestial,
                                               shape_out=wcs1.celestial.array_shape)
    v1 = reshape_wcs(vmap1,wcs1)

    if return_footprint:
        n1 = reshape_wcs(nmap1,wcs1)
        return v1,n1

    return v1

def modify_wcs_pixsize(wcs,pixsize):
    '''
    return wcs with different pixel size given with `pixsize`
    The number of data array would be estimated from original info
    '''
    header_orig = wcs.to_header()
    header = header_orig.copy()
    naxis = wcs.array_shape[::-1] # [RA,DEC,WAVE]

    if np.isclose(pixsize/3600,header['CDELT1']) and np.isclose(pixsize/3600,header['CDELT2']):
        print("WARNING:: same pixel size is required as input wcs.")
        return wcs,None

    header['CDELT1'] = pixsize/3600*np.sign(header_orig['CDELT1']) # [arcsec] --> [deg]
    header['CDELT2'] = pixsize/3600*np.sign(header_orig['CDELT2']) # [arcsec] --> [deg]

    pixsize_orig = header_orig['CDELT1']*3600
    header['NAXIS1'] = int((np.abs(pixsize_orig/pixsize)*naxis[0])+0.5)+2 # 1pix tolerance both sides
    header['NAXIS2'] = int((np.abs(pixsize_orig/pixsize)*naxis[1])+0.5)+2 # 1pix tolerance both sides
    if header['NAXIS1']%2==0:
        header['NAXIS1'] = header['NAXIS1']+1
    if header['NAXIS2']%2==0:
        header['NAXIS2'] = header['NAXIS2']+1

    header['CRPIX1'] = (header['NAXIS1']-1)/2 +1 # center pix
    header['CRPIX2'] = (header['NAXIS2']-1)/2 +1 # center pix

    if len(wcs.wcs.ctype) == 3:
        header['NAXIS3'] = 1

    return WCS(header),pixsize

def galactic_wcs(data,gl,gb,wcs,pixsize=None):
    '''
    return wcs for galactic coordinate from equatorial wcs
    data: data array defined with equatorial wcs
          GAL coord defined as square to include data
    gl,gb: galactic coordinate arrays with equatorial wcs
    wcs: equatorial wcs
    '''
    from astropy.io import fits
    ghdr = fits.PrimaryHDU(data=data).header

    is_cube = False
    if len(wcs.wcs.ctype) == 3:
        is_cube = True

    ghdr['CTYPE1'] = 'GLON-TAN' #gl
    ghdr['CTYPE2'] = 'GLAT-TAN' #gb

    if pixsize is None:
        ghdr['CDELT1'] = np.abs(wcs.wcs.cdelt[0])
        ghdr['CDELT2'] = np.abs(wcs.wcs.cdelt[1])
    else:
        ghdr['CDELT1'] = pixsize/3600
        ghdr['CDELT2'] = pixsize/3600

    crpix_coord = SkyCoord(ra=wcs.wcs.crval[0]*units.deg,
                           dec=wcs.wcs.crval[1]*units.deg,
                           frame='fk5')

    ghdr['CRVAL1'] = crpix_coord.galactic.l.deg
    ghdr['CRVAL2'] = crpix_coord.galactic.b.deg

    ss = ~np.isnan(data)
    gal_arrays1,ghdr['CRPIX1'] = misc.array2bins(gl[ss],ghdr['CDELT1'],ghdr['CRVAL1'])
    gal_arrays2,ghdr['CRPIX2'] = misc.array2bins(gb[ss],ghdr['CDELT2'],ghdr['CRVAL2'])

    ghdr['NAXIS1'] = len(gal_arrays1)
    ghdr['NAXIS2'] = len(gal_arrays2)

    if is_cube:
        ghdr['CTYPE3'] = wcs.wcs.ctype[2]
        ghdr['CDELT3'] = wcs.wcs.cdelt[2]
        ghdr['CRVAL3'] = wcs.wcs.crval[2]
        gal_arrays3,ghdr['CRPIX3'] = None,wcs.wcs.crpix[2]
        ghdr['NAXIS3'] = wcs.array_shape[0]

    return WCS(ghdr)

# functions for WCS world-coordinate-system

def reshape_wcs(x0,w0):
    '''
    modify array shape x0 from w0 to w1
    x0: input array which will be flatten before reshaping
    w0: WCS class. x0 will be reshaped to be fitted to w0
    '''
    sp = w0.array_shape # [WAVE,DEC,RA]
    return np.reshape(x0.flatten(), sp)

def modify_array_wcs(x0, w0, w1):
    '''
    modify array shape x0 from w0 to w1 wo wcs reprojection for speedup
    pixel size must be same btw wo and w1
    x0: array of [1][M][N]
    w0: WCS class of x0
    w1: target WCS class. x0 will be modified to be fitted to w1 from w0
    '''
    sp1 = w1.array_shape # [WAVE,DEC,RA]
    cp1 = w1.wcs.crpix[::-1] # [WAVE,DEC,RA]
    return modify_array(x0, w0, sp1, cp1)

def modify_array(x0,w0,sp1,cp1):
    '''
    modify array shape x0 from w0 to sp1,cp1 wo wcs reprojection for speedup
    x0: array of [1][M][N]
    w0: WCS class of x0
    sp1: array dimension to be returned (WAVE,DEC,RA). WAVE should be 1.
    cp1: reference center pixel position (WAVE,DEC,RA). WAVE should be 1.
    '''
    sp0 = w0.array_shape # [WAVE,DEC,RA]
    cp0 = w0.wcs.crpix[::-1] # [WAVE,DEC,RA]

    if len(sp0) == 2:
        sp0 = [1, sp0[0], sp0[1]]
        pass
    if len(cp0) == 2:
        cp0 = [1, cp0[0], cp0[1]]
        pass

    is_cube = True
    if len(sp1) == 2 and len(cp1) == 2: # 2d to be returned
        is_cube = False
        sp1 = [1, sp1[0], sp1[1]]
        cp1 = [1, cp1[0], cp1[1]]
        pass
    elif not (len(sp1) == 3 and len(cp1) == 3):
        print(f"WARNING:: Invalid array type. [sp1: array dimension to be returned]={len(sp1)} and [cp1: reference center pixel position array]={len(cp1)} is not same shape.")
        pass

    p = np.nan # value for extended pixels

    # RA
    i = 2
    n = int(cp1[i] - cp0[i])
    m = int(sp1[i] - sp0[i])-n

    if len(np.shape(x0)) == 2:
        #print(np.shape(x0))
        x0 = np.reshape(x0,(1,*np.shape(x0)))
        #print(np.shape(x0))
    if n>=0 and m>=0:
        x1 = np.insert(x0, [0]*n+[sp0[i]]*m, p, axis=i)
    elif n<0 and m<0:
        x1 = x0[:,:,-n:m]
    elif n>=0 and m<0:
        x1 = np.insert(x0, [0]*n, p, axis=i)[:,:,:m]
    elif n<0 and m>=0:
        x1 = np.insert(x0, [sp0[i]]*m, p, axis=i)[:,:,-n:]

    # DEC
    i = 1
    n = int(cp1[i] - cp0[i])
    m = int(sp1[i] - sp0[i])-n

    if n>=0 and m>=0:
        x1 = np.insert(x1, [0]*n+[sp0[i]]*m, p, axis=i)
    elif n<0 and m<0:
        x1 = x1[:,-n:m,:]
    elif n>=0 and m<0:
        x1 = np.insert(x1, [0]*n, p, axis=i)[:,:m,:]
    elif n<0 and m>=0:
        x1 = np.insert(x1, [sp0[i]]*m, p, axis=i)[:,-n:,:]

    if not is_cube:
        return x1[0]

    return x1


def meshgrid_wcs(w):
    '''
    return pixle-number meshgrid x,y (corresponds to RA,DEC) from wcs
    '''
    sp = w.array_shape
    if len(sp)==3:  # cube [WAVE,DEC,RA]
        return np.meshgrid(np.arange(sp[2]), np.arange(sp[1]))
    return np.meshgrid(np.arange(sp[1]), np.arange(sp[0])) # [DEC,RA]

