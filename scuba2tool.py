#!python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy
from scipy import stats
from scipy import signal
import copy

import healpy as hp

import astropy

from astropy import units
from astropy import convolution
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import reproject

mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'axes.facecolor': 'w'})
mpl.rcParams.update({'axes.edgecolor': 'k'})
mpl.rcParams.update({'figure.facecolor': 'w'})
mpl.rcParams.update({'figure.edgecolor': 'w'})
mpl.rcParams.update({'axes.grid': True})
mpl.rcParams.update({'grid.linestyle': ':'})
mpl.rcParams.update({'figure.figsize': [12, 9]})

# functions with SkyCoord

def get_tauAcent():
    cent = SkyCoord(ra=83.633083*astropy.units.deg,
                dec=22.014500*astropy.units.deg,
                frame='fk5')
    return cent

def get_radec(coord):
    return (coord.ra.deg, coord.dec.deg)

def get_glgb(coord):
    return (coord.galactic.l.deg, coord.galactic.b.deg)

# functions for SCUBA2 values

def get_fcfbeam():
    return 495*1.35*1e3 # [pW] --> [mJy/arcsec2]

def get_fcf():
    return 2.07*1.35*1e3 # [pW] --> [mJy/arcsec2]

def get_beamfwhm():
    return 13.6 #[arcsec]

def get_beamarea():
    return (2*np.pi*(get_beamfwhm())**2)/(8*np.log(2)) #[arcsec2]

def nika_get_fcf():
    return 1./(2*np.pi*(18.2/2.355)**2)*1000/1.28 # [Jy/beam] to [mJy/arcsec2]

# functions for polarization calculations

def polamp(q, u, dq=0, du=0):
    '''
    return polarization amplitude of sqrt(Q**2 + U**2)
    '''
    ret = q**2 + u**2 - dq**2 - du**2
    ret[ret<0] = 0
    return np.sqrt(ret)

def polfrac(i, q, u, dq=0, du=0):
    '''
    return polarization fraction of sqrt(Q**2 + U**2)/I
    '''
    return polamp(q, u, dq, du)/i

def polang(q, u):
    '''
    return polarization angle [deg] with correction for IAU
    '''
    ret = 0.5*np.arctan2(u,q)*180/np.pi
    ret[ret < 0] = ret[ret < 0]+180
    return ret

def dpolamp(q, u, dq, du, simple=False):
    '''
    return rms of polarization amplitude
    '''
    if simple:
        ip = polamp(q,u)
    else:
        ip = polamp(q,u,dq,du)
    return np.sqrt(q**2*dq**2 + u**2*du**2)/ip

def dpolfrac(i, q, u, di, dq, du, simple=False):
    '''
    return rms of polarization fraction
    '''
    if simple:
        p = polfrac(i, q, u)
    else:
        p = polfrac(i, q, u, dq, du)
    return np.sqrt((q**2)*(dq**2) + (u**2)*(du**2) + (p**4)*(i**2)*(di**2))/(p*(i**2))

def dpolang(q, u, dq, du, simple=False):
    '''
    return rms of polarization angle [deg]
    '''
    if simple:
        ip = polamp(q,u)
    else:
        ip = polamp(q,u,dq,du)
    return np.sqrt(q**2*du**2 + u**2*dq**2)/(2*ip**2)*180/np.pi

def diqu(di, dq, du):
    '''
    return total rms of I, Q, U
    '''
    return dq*du*di/np.sqrt((di*dq)**2+(dq*du)**2+(du*di)**2)

def parallactic_angle(ra, dec):  # radec [deg]
    '''
    return parallactic angle [deg] of EQ --> GAL
    '''
    R_coord = hp.Rotator(coord=['C', 'G'])
    sp = np.shape(ra)
    ze = np.array([180./2-dec.flatten(), ra.flatten()])*np.pi/180
    psi = R_coord.angle_ref(ze)
    return np.reshape(psi, sp)*180/np.pi

def galactic_iqu(i, q, u, psi):
    '''
    return IQU in GAL coordinate from EQ IQU by using parallactic angle [deg]
    '''
    newi = i
    newq = q * np.cos(2*psi/180*np.pi) + u * np.sin(2*psi/180*np.pi)
    newu = -q * np.sin(2*psi/180*np.pi) + u * np.cos(2*psi/180*np.pi)
    return newi, newq, newu

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

    ghdr['CRVAL1'],ghdr['CRVAL2'] = get_glgb(crpix_coord)

    ss = ~np.isnan(data)
    gal_arrays1,ghdr['CRPIX1'] = array2bins(gl[ss],ghdr['CDELT1'],ghdr['CRVAL1'])
    gal_arrays2,ghdr['CRPIX2'] = array2bins(gb[ss],ghdr['CDELT2'],ghdr['CRVAL2'])

    ghdr['NAXIS1'] = len(gal_arrays1)
    ghdr['NAXIS2'] = len(gal_arrays2)

    if is_cube:
        ghdr['CTYPE3'] = wcs.wcs.ctype[2]
        ghdr['CDELT3'] = wcs.wcs.cdelt[2]
        ghdr['CRVAL3'] = wcs.wcs.crval[2]
        gal_arrays3,ghdr['CRPIX3'] = None,wcs.wcs.crpix[2]
        ghdr['NAXIS3'] = wcs.array_shape[0]

    return WCS(ghdr)

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

# plotters

def wcs_subplots(wcs=None, nrows=1, ncols=1, **fig_kw):
    if 'figsize' not in fig_kw:
        fig_kw['figsize'] = (7.4*ncols,6*nrows)

    fig = plt.figure(**fig_kw)
    ax = [[nrows,ncols,i+1] for i in np.arange(nrows*ncols)]

    if wcs is not None:
        for i in range(len(ax)):
            ax[i] = fig.add_subplot(nrows, ncols, i+1, projection=wcs.celestial)
            ax[i].set_facecolor('grey')

    if len(ax)==1:
        ax = ax[0]

    return fig,ax

from astropy.visualization.wcsaxes import SphericalCircle
def wcsaxes_circle(xy, r, ax, transform, **kwargs):
    if not isinstance(xy,units.quantity.Quantity):
        xy = xy*units.degree
    if not isinstance(r,units.quantity.Quantity):
        r = r*units.degree
    if isinstance(transform,str):
        transform = ax.get_transform(transform)

    circ = SphericalCircle(xy,r,transform=transform,**kwargs)
    ax.add_patch(circ)
    return ax

def wcsaxes_point(xy, ax, transform, **kwargs):
    if not isinstance(xy,units.quantity.Quantity):
        xy = xy*units.degree
    if isinstance(transform,str):
        transform = ax.get_transform(transform)

    ax.scatter(*xy, transform=transform,  **kwargs)
    return ax

def wcsaxes_lim(xlim,ylim,ax,wcs):
    xl, yl = wcs.celestial.all_pix2world(ax.get_xlim()*units.degree, ax.get_ylim()*units.degree, 1)
    if xlim[0] is None:
        xlim[0] = xl[0]
    if xlim[1] is None:
        xlim[1] = xl[1]
    if ylim[0] is None:
        ylim[0] = yl[0]
    if ylim[1] is None:
        ylim[1] = yl[1]
    xlim_pix, ylim_pix = wcs.celestial.all_world2pix(xlim*units.degree, ylim*units.degree, 1)
    ax.set_xlim(*xlim_pix)
    ax.set_ylim(*ylim_pix)
    return ax

def get_axeslim_nika(frame):
    max_point = [83.56*units.degree,22.08*units.degree] #[deg] in FK5
    min_point = [83.71*units.degree,21.95*units.degree] #[deg] in FK5
    max_coord = SkyCoord(ra = max_point[0],
                         dec= max_point[1],
                         frame='fk5')
    min_coord = SkyCoord(ra = min_point[0],
                         dec= min_point[1],
                         frame='fk5')
    if frame.lower()[:3]=='fk5':
        xlim = [min_coord.ra.deg,max_coord.ra.deg]
        ylim = [min_coord.dec.deg,max_coord.dec.deg]
    elif frame.lower()[:3]=='gal':
        xlim = [min_coord.galactic.l.deg,max_coord.galactic.l.deg]
        ylim = [min_coord.galactic.b.deg,max_coord.galactic.b.deg]
    else:
        print(f"ERROR:: Invalid frame. \"{frame}\".")
        pass
    return xlim,ylim

def get_axeslim_scuba2(frame):
    max_point = [83.47*units.degree,22.17*units.degree] #[deg] in FK5
    min_point = [83.8*units.degree,21.85*units.degree] #[deg] in FK5
    max_coord = SkyCoord(ra = max_point[0],
                         dec= max_point[1],
                         frame='fk5')
    min_coord = SkyCoord(ra = min_point[0],
                         dec= min_point[1],
                         frame='fk5')
    if frame.lower()[:3]=='fk5':
        xlim = [min_coord.ra.deg,max_coord.ra.deg]
        ylim = [min_coord.dec.deg,max_coord.dec.deg]
    elif frame.lower()[:3]=='gal':
        xlim = [min_coord.galactic.l.deg,max_coord.galactic.l.deg]
        ylim = [min_coord.galactic.b.deg,max_coord.galactic.b.deg]
    else:
        print(f"ERROR:: Invalid frame. \"{frame}\".")
        pass
    return xlim,ylim

def plotmap(x, wcs, title='', fig=None, ax=None,
            ax_decimal=False, ax_galactic=False, ax_fk5=False,count=None, plot_contour=False,
            xlim=(None, None), ylim=(None, None), nikalim=False, scuba2lim=False,
            show_colorscale=True, cmap='',**kwargs):
    '''
    easy map plotter function
    '''
    if fig is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111,projection=wcs.celestial)
        ax.set_facecolor('grey')
    elif hasattr(ax, "__iter__") and len(ax)==3:
        ax = fig.add_subplot(*ax, projection=wcs.celestial)
        ax.set_facecolor('grey')

    ax.set_aspect('equal')
    cax = None
    divider = make_axes_locatable(ax)
    npad = 0.1
    if ax_galactic or ax_fk5:
        npad = 1
        ax.set_title(title,y=1.15)
    else:
        ax.set_title(title)
    cax0 = divider.append_axes("top", size="5%", pad=npad) # to keep the title space
    cax0.set_axis_off()
    cax1 = divider.append_axes("bottom", size="6%", pad=npad) # to keep the title space
    cax1.set_axis_off()
    cax1 = divider.append_axes("left", size="4%", pad=npad) # to keep the title space
    cax1.set_axis_off()
    if show_colorscale:
        #if ax_galactic or ax_fk5:
        #    npad = 1
        cax = divider.append_axes("right", size="5%", pad=npad, axes_class=mpl.axes.Axes)

    overlay = None
    if ax_galactic:
        overlay = ax.get_coords_overlay('galactic')
        overlay[0].set_axislabel('GL', color='b', size='small')
        overlay[1].set_axislabel('GB', color='b', size='small')
    if ax_fk5:
        overlay = ax.get_coords_overlay('fk5')
        overlay[0].set_axislabel('RA', color='b', size='small')
        overlay[1].set_axislabel('DEC', color='b', size='small')
    if overlay is not None:
        overlay.grid(color='skyblue')
        overlay[0].tick_params(labelcolor='b', labelsize='small')
        overlay[1].tick_params(labelcolor='b', labelsize='small')

    if ax_decimal:
        ax.coords[0].set_format_unit(units.degree, decimal=True, show_decimal_unit=False)
        ax.coords[1].set_format_unit(units.degree, decimal=True, show_decimal_unit=False)
        if overlay is not None:
            overlay[0].set_format_unit(units.degree, decimal=True, show_decimal_unit=False)
            overlay[1].set_format_unit(units.degree, decimal=True, show_decimal_unit=False)
        pass

    if cmap == '':
        cmap = copy.copy(plt.get_cmap('turbo', count))
        cmap.set_bad('grey')
        cmap.set_under('k')
    else:
        cmap = copy.copy(plt.get_cmap(cmap, count))
    #cmap.set_over('w')

    if len(np.shape(x)) == 3: # cube
        xmap = x[0]
    else:
        xmap = x

    if plot_contour:
        im = ax.contour(xmap, origin='lower', cmap=cmap, **kwargs)
        if cax is not None:
            fig.colorbar(im, extend='min', cax=cax)
    else:
        im = ax.imshow(xmap, origin='lower', cmap=cmap, **kwargs)
        if cax is not None:
            fig.colorbar(im, extend='min', cax=cax)

    if nikalim:
        if wcs.axis_type_names[0] == 'RA':
            xlim,ylim = get_axeslim_nika('fk5')
            wcsaxes_lim(xlim,ylim,ax,wcs)
        elif wcs.axis_type_names[0] == 'GLON':
            xlim,ylim = get_axeslim_nika('gal')
            wcsaxes_lim(xlim,ylim,ax,wcs)
        else:
            print(f"ERROR:: Unknown axis type name in this wcs : {wcs.axis_type_names}")
            pass

    if scuba2lim:
        if wcs.axis_type_names[0] == 'RA':
            xlim,ylim = get_axeslim_scuba2('fk5')
            wcsaxes_lim(xlim,ylim,ax,wcs)
        elif wcs.axis_type_names[0] == 'GLON':
            xlim,ylim = get_axeslim_scuba2('gal')
            wcsaxes_lim(xlim,ylim,ax,wcs)
        else:
            print(f"ERROR:: Unknown axis type name in this wcs : {wcs.axis_type_names}")
            pass

    if not (xlim[0] is None and xlim[1] is None and ylim[0] is None and ylim[1] is None):
        xlim = list(xlim)
        ylim = list(ylim)
        wcsaxes_lim(xlim,ylim,ax,wcs)
        pass

    return fig, ax

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

# fits file treatments

def get_scuba2map(fn, prior="", ext='', **kwargs):
    '''
    return scuba2map class from directory name
    - prior: need to input if subset IQUmap to be registered. filename will be = {fn}/maps/{prior}_{I,Q,U}map.fits

    hints of kwargs:
    - imap: registered automatically in this function if not given
    - qmap: registered automatically in this function if not given
    - umap: registered automatically in this function if not given
    - iautomap: registered automatically in this function if not given
    - wcs_target: reference WCS
    '''
    from astropy.io import fits

    if fn == '':
        fn = './'
        pass
    if fn[-1] != '/':
        fn += '/'

    if 'imap' not in kwargs:
        if prior != "":
            kwargs['imap'] = fits.open(fn+"/maps/"+prior+"_Imap.fits")
        else:
            kwargs['imap'] = fits.open(fn+"iext"+ext+".fits")
        pass
    if 'qmap' not in kwargs:
        if prior != "":
            kwargs['qmap'] = fits.open(fn+"/maps/"+prior+"_Qmap.fits")
        else:
            kwargs['qmap'] = fits.open(fn+"qext"+ext+".fits")
        pass
    if 'umap' not in kwargs:
        if prior != "":
            kwargs['umap'] = fits.open(fn+"/maps/"+prior+"_Umap.fits")
        else:
            kwargs['umap'] = fits.open(fn+"uext"+ext+".fits")
        pass
    if 'iautomap' not in kwargs:
        fname = fn+"/maps/"+prior+"_imap.fits" if prior != "" else fn+"iauto"+ext+".fits"
        try:
            kwargs['iautomap'] = fits.open(fname)
        except:
            print(f"WARNING:: iautomap cannot be registered \"{fname}\" --> skipped.")
        pass

    if 'astmask' not in kwargs and os.path.isfile(fn+"astmask"+ext+".fits"):
        kwargs['astmask'] = fn+"astmask"+ext+".fits"
    if 'astmask' in kwargs and kwargs['astmask'] is not None:
        try:
            if type(kwargs['astmask']) is str:
                kwargs['astmask'] = fits.open(kwargs['astmask'])
        except:
            print(f"ERROR:: Cannot register astmaskmap of \"{kwargs['astmask']}\"")

    if 'pcamask' not in kwargs and os.path.isfile(fn+"pcamask"+ext+".fits"):
        kwargs['pcamask'] = fn+"pcamask"+ext+".fits"
    if 'pcamask' in kwargs and kwargs['pcamask'] is not None:
        try:
            if type(kwargs['pcamask']) is str:
                kwargs['pcamask'] = fits.open(kwargs['pcamask'])
        except:
            print(f"ERROR:: Cannot register pcamaskmap of \"{kwargs['pcamask']}\"")

    return scuba2map(**kwargs)

# scuba2map class

class scuba2map():
    def __init__(self, imap, qmap, umap, fcf=1.,
                 iautomap=None, wcs_target=None, data_target=None,
                 noise_kind=None, pixsize=None, pisize_gal=None, filtsize=None,
                 astmask=None, pcamask=None, nanvalue=None, set_zeronan=False):
        # fcf: conversion factor for input fits data arrays (recommendation is fcf to get arrays in [mJy/arcsec2])

        import warnings
        from astropy.wcs import FITSFixedWarning
        warnings.filterwarnings('ignore', message="'datfix' made the change",
                                category=FITSFixedWarning)
        warnings.filterwarnings('ignore', message="'obsfix' made the change",
                                category=FITSFixedWarning)

        iwcs = WCS(imap[0].header)
        qwcs = WCS(qmap[0].header)
        uwcs = WCS(umap[0].header)

        if set_zeronan and nanvalue is None:
            nanvalue = 0

        if wcs_target is None:
            wcs_target = iwcs
        else:
            if data_target is None:
                print("WARNING:: WCS target is assigned but not data target. Gal coordinate could be odd...")

        if pixsize is not None:
            wcs_target,pixsize = modify_wcs_pixsize(wcs_target,pixsize)

        x, y  = meshgrid_wcs(wcs_target)
        world = wcs_target.celestial.wcs_pix2world(x, y, 1) # [WAVE,RA,DEC] --> [RA,DEC] with celestial
        coord = SkyCoord(ra=world[0]*units.deg, dec=world[1]*units.deg, frame='fk5')

        # equatorial coord
        ra  = reshape_wcs(coord.ra.deg,  wcs_target)
        dec = reshape_wcs(coord.dec.deg, wcs_target)

        # noise calc
        if len(imap)==1:
            noise_kind = None
            di_raw = np.zeros_like(imap[0].data)*np.nan
            dq_raw = np.zeros_like(qmap[0].data)*np.nan
            du_raw = np.zeros_like(umap[0].data)*np.nan
        else:
            if noise_kind is None:
                noise_kind = imap[1].header['EXTNAME'] if 'EXTNAME' in imap[1].header else 'std'
            noise_kind = noise_kind.lower()
            if 'var' in noise_kind or 'variance' in noise_kind:
                di_raw = np.sqrt(imap[1].data)
                dq_raw = np.sqrt(qmap[1].data)
                du_raw = np.sqrt(umap[1].data)
                pass
            else:
                di_raw = imap[1].data
                dq_raw = qmap[1].data
                du_raw = umap[1].data
                if not ('std' in noise_kind or 'deviation' in noise_kind):
                    print(f"WARNING:: Invalid noise_kind: \"{noise_kind}\". Treated as standard deviation.")
                pass

        # filtering
        if filtsize is not None:
            if np.isclose(filtsize,0.0):
                print("WARNING:: filtering stddev (filtsize) is set to zero. filtsize should be >0 [arcsec].")
            pixsize_filt = iwcs.proj_plane_pixel_scales()[0].value*3600 #[arcsec]
            i_raw = filter_gaussian(imap[0].data,pixsize_filt,filtsize)
            q_raw = filter_gaussian(qmap[0].data,pixsize_filt,filtsize)
            u_raw = filter_gaussian(umap[0].data,pixsize_filt,filtsize)
            di_raw = filter_gaussian(di_raw,pixsize_filt,filtsize)
            dq_raw = filter_gaussian(dq_raw,pixsize_filt,filtsize)
            du_raw = filter_gaussian(du_raw,pixsize_filt,filtsize)
        else:
            i_raw = imap[0].data
            q_raw = qmap[0].data
            u_raw = umap[0].data

        # data assignment
        if pixsize is None and np.isclose(iwcs.proj_plane_pixel_scales()[0].value,wcs_target.proj_plane_pixel_scales()[0].value):
            i = modify_array_wcs(i_raw,iwcs,wcs_target)*fcf
            q = modify_array_wcs(q_raw,qwcs,wcs_target)*fcf
            u = modify_array_wcs(u_raw,uwcs,wcs_target)*fcf
            di = modify_array_wcs(di_raw,iwcs,wcs_target)*fcf
            dq = modify_array_wcs(dq_raw,qwcs,wcs_target)*fcf
            du = modify_array_wcs(du_raw,uwcs,wcs_target)*fcf
        else:
            i = wcs_reproject(i_raw,iwcs,wcs_target)*fcf
            q = wcs_reproject(q_raw,qwcs,wcs_target)*fcf
            u = wcs_reproject(u_raw,uwcs,wcs_target)*fcf
            di = wcs_reproject(di_raw,iwcs,wcs_target)*fcf
            dq = wcs_reproject(dq_raw,qwcs,wcs_target)*fcf
            du = wcs_reproject(du_raw,uwcs,wcs_target)*fcf

        if nanvalue is not None:
            i[i==nanvalue] = np.nan
            q[q==nanvalue] = np.nan
            u[u==nanvalue] = np.nan
            di[di==nanvalue] = np.nan
            dq[dq==nanvalue] = np.nan
            du[du==nanvalue] = np.nan

        ## iauto map
        iauto  = None
        diauto = None
        if iautomap is not None:
            iautowcs = WCS(iautomap[0].header)
            iauto  = modify_array_wcs(iautomap[0].data,iautowcs,wcs_target)*fcf
            diauto = modify_array_wcs(np.sqrt(iautomap[1].data),iautowcs,wcs_target)*fcf

        ## astmask
        am = None
        if astmask is not None:
            maskwcs = WCS(astmask[0].header)
            am  = modify_array_wcs(astmask[0].data,maskwcs,wcs_target)

        ## pcamask
        pm = None
        if pcamask is not None:
            maskwcs = WCS(pcamask[0].header)
            pm  = modify_array_wcs(pcamask[0].data,maskwcs,wcs_target)

        # galactic coord
        gall = reshape_wcs(coord.galactic.l.deg, wcs_target)
        galb = reshape_wcs(coord.galactic.b.deg, wcs_target)
        psi = parallactic_angle(coord.ra.deg, coord.dec.deg)

        self.header = imap[0].header
        self.wcs = wcs_target
        self.x = x
        self.y = y

        self.ra = ra
        self.dec = dec

        self.i = i
        self.q = q
        self.u = u
        self.di = di
        self.dq = dq
        self.du = du

        self.iauto = iauto
        self.diauto = diauto

        self.pcamask = pm
        self.astmask = am

        self.gl = gall
        self.gb = galb
        self.gi,self.gq,self.gu = galactic_iqu(i,q,u,psi)
        self.gdi,self.gdq,self.gdu = galactic_iqu(di,dq,du,psi)
        self.psi = psi

        self._gal = None
        self._galpix = pisize_gal
        self._galrefdata = data_target

        return

    @property
    def gal(self):
        if self._gal is None:
            self._gal = scuba2galmap(self)
            pass
        return self._gal

    @property
    def iqu(self):
        return (self.i, self.q, self.u)

    @property
    def diqu(self):
        return (self.di, self.dq, self.du)

    @property
    def radec(self):
        return (self.ra, self.dec)

    @property
    def pi(self):
        return polamp(self.q,self.u,self.dq,self.du)

    @property
    def pf(self):
        return polfrac(self.i,self.q,self.u,self.dq,self.du)

    @property
    def pa(self):
        return polang(self.q,self.u)

    @property
    def pixarea_arcsec2(self):
        return self.pixarea_deg2*3600*3600

    @property
    def pixarea_deg2(self):
        return self.wcs.proj_plane_pixel_area().value

    @property
    def pixscales_arcsec(self):
        return [x*3600 for x in self.pixscales_deg]

    @property
    def pixscales_deg(self):
        return [x.value for x in self.wcs.proj_plane_pixel_scales()]

    pass

class scuba2galmap():
    def __init__(self, eqmap): # refdata should be same shape as eqmap

        import warnings
        from astropy.wcs import FITSFixedWarning
        warnings.filterwarnings('ignore', message="'datfix' made the change",
                                category=FITSFixedWarning)
        warnings.filterwarnings('ignore', message="'obsfix' made the change",
                                category=FITSFixedWarning)

        eqmap_gali,  eqmap_galq,  eqmap_galu  = galactic_iqu(eqmap.i,  eqmap.q,  eqmap.u,  eqmap.psi)
        eqmap_galdi, eqmap_galdq, eqmap_galdu = galactic_iqu(eqmap.di, eqmap.dq, eqmap.du, eqmap.psi)

        refdata = eqmap._galrefdata
        pixsize = eqmap._galpix

        if refdata is None:
            refdata = eqmap_gali

        wcs_target = galactic_wcs(refdata, eqmap.gl, eqmap.gb, eqmap.wcs, pixsize=pixsize)
        fact = np.sqrt(wcs_target.proj_plane_pixel_area().value/eqmap.wcs.proj_plane_pixel_area().value)

        i = wcs_reproject(eqmap_gali,eqmap.wcs,wcs_target)
        q = wcs_reproject(eqmap_galq,eqmap.wcs,wcs_target)
        u = wcs_reproject(eqmap_galu,eqmap.wcs,wcs_target)

        di = wcs_reproject(eqmap_galdi,eqmap.wcs,wcs_target)*fact
        dq = wcs_reproject(eqmap_galdq,eqmap.wcs,wcs_target)*fact
        du = wcs_reproject(eqmap_galdu,eqmap.wcs,wcs_target)*fact

        x, y  = meshgrid_wcs(wcs_target)
        world = wcs_target.celestial.wcs_pix2world(x, y, 1) # [WAVE,RA,DEC] --> [RA,DEC] with celestial
        coord = SkyCoord(l=world[0]*units.deg, b=world[1]*units.deg, frame='galactic')

        # galactic coord
        gl = reshape_wcs(coord.l.deg, wcs_target)
        gb = reshape_wcs(coord.b.deg, wcs_target)

        self.wcs = wcs_target
        self.x = x
        self.y = y

        self.gl = gl
        self.gb = gb

        self.i = i
        self.q = q
        self.u = u

        self.di = di
        self.dq = dq
        self.du = du

    @property
    def iqu(self):
        return (self.i, self.q, self.u)

    @property
    def diqu(self):
        return (self.di, self.dq, self.du)

    @property
    def glgb(self):
        return (self.gl, self.gb)

    @property
    def pi(self):
        return polamp(self.q,self.u,self.dq,self.du)

    @property
    def pf(self):
        return polfrac(self.i,self.q,self.u,self.dq,self.du)

    @property
    def pa(self):
        return polang(self.q,self.u)

    @property
    def pixarea_arcsec2(self):
        return self.pixarea_deg2*3600*3600

    @property
    def pixarea_deg2(self):
        return self.wcs.proj_plane_pixel_area().value

    @property
    def pixscales_arcsec(self):
        return [x*3600 for x in self.pixscales_deg]

    @property
    def pixscales_deg(self):
        return [x.value for x in self.wcs.proj_plane_pixel_scales()]

    pass
