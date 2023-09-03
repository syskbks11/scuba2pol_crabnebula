#!python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy
from scipy import stats
from scipy import signal
import copy

import astropy

from astropy import units
from astropy import convolution
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import reproject

import wcstool
import polcalc
import misc

# functions with SkyCoord

def get_tauAcent():
    cent = SkyCoord(ra=83.633083*astropy.units.deg,
                dec=22.014500*astropy.units.deg,
                frame='fk5')
    return cent

# functions for SCUBA2 POL

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

# for analysis with NIKA data combination

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
            wcs_target,pixsize = wcstool.modify_wcs_pixsize(wcs_target,pixsize)

        x, y  = wcstool.meshgrid_wcs(wcs_target)
        world = wcs_target.celestial.wcs_pix2world(x, y, 1) # [WAVE,RA,DEC] --> [RA,DEC] with celestial
        coord = SkyCoord(ra=world[0]*units.deg, dec=world[1]*units.deg, frame='fk5')

        # equatorial coord
        ra  = wcstool.reshape_wcs(coord.ra.deg,  wcs_target)
        dec = wcstool.reshape_wcs(coord.dec.deg, wcs_target)

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
            i_raw = misc.filter_gaussian(imap[0].data,pixsize_filt,filtsize)
            q_raw = misc.filter_gaussian(qmap[0].data,pixsize_filt,filtsize)
            u_raw = misc.filter_gaussian(umap[0].data,pixsize_filt,filtsize)
            di_raw = misc.filter_gaussian(di_raw,pixsize_filt,filtsize)
            dq_raw = misc.filter_gaussian(dq_raw,pixsize_filt,filtsize)
            du_raw = misc.filter_gaussian(du_raw,pixsize_filt,filtsize)
        else:
            i_raw = imap[0].data
            q_raw = qmap[0].data
            u_raw = umap[0].data

        # data assignment
        if pixsize is None and np.isclose(iwcs.proj_plane_pixel_scales()[0].value,wcs_target.proj_plane_pixel_scales()[0].value):
            i = wcstool.modify_array_wcs(i_raw,iwcs,wcs_target)*fcf
            q = wcstool.modify_array_wcs(q_raw,qwcs,wcs_target)*fcf
            u = wcstool.modify_array_wcs(u_raw,uwcs,wcs_target)*fcf
            di = wcstool.modify_array_wcs(di_raw,iwcs,wcs_target)*fcf
            dq = wcstool.modify_array_wcs(dq_raw,qwcs,wcs_target)*fcf
            du = wcstool.modify_array_wcs(du_raw,uwcs,wcs_target)*fcf
        else:
            i = wcstool.wcs_reproject(i_raw,iwcs,wcs_target)*fcf
            q = wcstool.wcs_reproject(q_raw,qwcs,wcs_target)*fcf
            u = wcstool.wcs_reproject(u_raw,uwcs,wcs_target)*fcf
            di = wcstool.wcs_reproject(di_raw,iwcs,wcs_target)*fcf
            dq = wcstool.wcs_reproject(dq_raw,qwcs,wcs_target)*fcf
            du = wcstool.wcs_reproject(du_raw,uwcs,wcs_target)*fcf

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
            iauto  = wcstool.modify_array_wcs(iautomap[0].data,iautowcs,wcs_target)*fcf
            diauto = wcstool.modify_array_wcs(np.sqrt(iautomap[1].data),iautowcs,wcs_target)*fcf

        ## astmask
        am = None
        if astmask is not None:
            maskwcs = WCS(astmask[0].header)
            am  = wcstool.modify_array_wcs(astmask[0].data,maskwcs,wcs_target)

        ## pcamask
        pm = None
        if pcamask is not None:
            maskwcs = WCS(pcamask[0].header)
            pm  = wcstool.modify_array_wcs(pcamask[0].data,maskwcs,wcs_target)

        # galactic coord
        gall = wcstool.reshape_wcs(coord.galactic.l.deg, wcs_target)
        galb = wcstool.reshape_wcs(coord.galactic.b.deg, wcs_target)
        psi = polcalc.parallactic_angle(coord.ra.deg, coord.dec.deg)

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
        self.gi,self.gq,self.gu = polcalc.galactic_iqu(i,q,u,psi)
        self.gdi,self.gdq,self.gdu = polcalc.galactic_iqu(di,dq,du,psi)
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
        return polcalc.polamp(self.q,self.u,self.dq,self.du)

    @property
    def pf(self):
        return polcalc.polfrac(self.i,self.q,self.u,self.dq,self.du)

    @property
    def pa(self):
        return polcalc.polang(self.q,self.u)

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

        eqmap_gali,  eqmap_galq,  eqmap_galu  = polcalc.galactic_iqu(eqmap.i,  eqmap.q,  eqmap.u,  eqmap.psi)
        eqmap_galdi, eqmap_galdq, eqmap_galdu = polcalc.galactic_iqu(eqmap.di, eqmap.dq, eqmap.du, eqmap.psi)

        refdata = eqmap._galrefdata
        pixsize = eqmap._galpix

        if refdata is None:
            refdata = eqmap_gali

        wcs_target = wcstool.galactic_wcs(refdata, eqmap.gl, eqmap.gb, eqmap.wcs, pixsize=pixsize)
        fact = np.sqrt(wcs_target.proj_plane_pixel_area().value/eqmap.wcs.proj_plane_pixel_area().value)

        i = wcstool.wcs_reproject(eqmap_gali,eqmap.wcs,wcs_target)
        q = wcstool.wcs_reproject(eqmap_galq,eqmap.wcs,wcs_target)
        u = wcstool.wcs_reproject(eqmap_galu,eqmap.wcs,wcs_target)

        di = wcstool.wcs_reproject(eqmap_galdi,eqmap.wcs,wcs_target)*fact
        dq = wcstool.wcs_reproject(eqmap_galdq,eqmap.wcs,wcs_target)*fact
        du = wcstool.wcs_reproject(eqmap_galdu,eqmap.wcs,wcs_target)*fact

        x, y  = wcstool.meshgrid_wcs(wcs_target)
        world = wcs_target.celestial.wcs_pix2world(x, y, 1) # [WAVE,RA,DEC] --> [RA,DEC] with celestial
        coord = SkyCoord(l=world[0]*units.deg, b=world[1]*units.deg, frame='galactic')

        # galactic coord
        gl = wcstool.reshape_wcs(coord.l.deg, wcs_target)
        gb = wcstool.reshape_wcs(coord.b.deg, wcs_target)

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
        return polcalc.polamp(self.q,self.u,self.dq,self.du)

    @property
    def pf(self):
        return polcalc.polfrac(self.i,self.q,self.u,self.dq,self.du)

    @property
    def pa(self):
        return polcalc.polang(self.q,self.u)

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
