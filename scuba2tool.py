#!python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
from scipy import stats

import healpy as hp

import astropy

from astropy import units
from astropy.coordinates import SkyCoord

mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'axes.facecolor': 'w'})
mpl.rcParams.update({'axes.edgecolor': 'k'})
mpl.rcParams.update({'figure.facecolor': 'w'})
mpl.rcParams.update({'figure.edgecolor': 'w'})
mpl.rcParams.update({'axes.grid': True})
mpl.rcParams.update({'grid.linestyle': ':'})
mpl.rcParams.update({'figure.figsize': [12, 9]})

def get_tauAcent():
    cent = SkyCoord(ra=83.633083*astropy.units.deg,
                dec=22.014500*astropy.units.deg,
                frame='fk5')
    return cent

def get_radec(coord):
    return (coord.ra.deg, coord.dec.deg)

def get_glgb(coord):
    return (coord.galactic.l.deg, coord.galactic.b.deg)

def get_fcfbeam():
    return 495*1.35*1e3 # [pW] --> [mJ/arcsec2]

def get_fcf():
    return 2.07*1.35*1e3 # [pW] --> [mJ/arcsec2]

def get_beamarea():
    return (2*np.pi*(14.6)**2)/(8*np.log(2)) #[arcsec2]

def header2array(d, v, r, n):
    '''
    return axis array from header information
    '''
    ret = np.zeros(n)
    ret[int(r)] = v
    ret[:int(r)] = v - d*np.arange(int(r), 0, -1)
    ret[int(r):] = v + d*np.arange(0, int(n-r), 1)
    return ret

def array2bins(x, dx=None, return_dx=False):
    '''
    return binning array from axis array
    '''
    if dx is None:
        dx = np.diff(x)
        dx = np.nanmedian(dx[np.abs(dx) > 1e-6])
    nbin = np.abs(int((np.nanmax(x)-np.nanmin(x)+dx)/dx))
    if return_dx:
        return np.linspace(np.nanmin(x)-dx/2, np.nanmax(x)+dx/2, nbin), dx
    else:
        return np.linspace(np.nanmin(x)-dx/2, np.nanmax(x)+dx/2, nbin)

def polamp(q, u, dq=0, du=0):
    '''
    return polarization amplitude of sqrt(Q**2 + U**2)
    '''
    return np.sqrt(q**2 + u**2 - dq**2 - du**2)

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
    ze = np.array([180./2-dec, ra])*np.pi/180
    psi = R_coord.angle_ref(ze)
    return psi*180/np.pi

def galactic_iqu(i, q, u, psi):
    '''
    return IQU in GAL coordinate from EQ IQU by using parallactic angle [deg]
    '''
    newi = i
    newq = q * np.cos(2*psi/180*np.pi) + u * np.sin(2*psi/180*np.pi)
    newu = -q * np.sin(2*psi/180*np.pi) + u * np.cos(2*psi/180*np.pi)
    return newi, newq, newu

def mask(x,mask):
    '''
    return masked x array with mask array
    (masked element will be replaced with np.nan)
    mask: 1 --> masked, 0 --> not masked
    '''
    ret = np.copy(x)
    ret[mask>0.001] = np.nan
    return ret

def plotcontour(xy, z, bins, title='', fig=None, ax=None, **kwargs):
    return plotmap(xy, z, bins, title=title, fig=fig, ax=ax, plot_contour=True, **kwargs)

def plotmap(xy, z, bins, title='', fig=None, ax=None, count=None, plot_contour=False, **kwargs):
    '''
    easy map plotter function
    xy: array of (2,N) xy[0] = x, xy[1] = y
    z: array to be mapped
    bins: array of binning for xy (2,M) bins[0] = binning for x, bins[1] = binning for y
    count: number of colors divided in color scale
    plot_contour: contour() is called if True, otherwise imshow() is called
                  if True, good to add more options like levels=[10,5,3] and colors=['k','grey','w']
    '''
    x = xy[0]
    y = xy[1]
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ss = ~np.isnan(z)
    ret = stats.binned_statistic_2d(x[ss], y[ss], z[ss], statistic='mean', bins=bins)
    cmap = None
    if count is not None:
        cmap = mpl.cm.get_cmap("jet", count)
        pass
    if not plot_contour:
        im = ax.imshow(ret.statistic.T, origin='lower',
                       extent=(bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]),
                       cmap=cmap, **kwargs)
        pass
    else:
        im = ax.contour(ret.statistic.T, origin='lower',
                        extent=(bins[0][0], bins[0][-1], bins[1][0], bins[1][-1]),
                        **kwargs)
        pass
    ax.set_title(title)
    ax.set_aspect('equal')

    if not plot_contour:
        fig.colorbar(im, ax=ax)
        pass

    return ret

def get_catmap(fn, catmap_filename="catnominal.FIT"):
    '''
    return catmap fits class from directory name
    '''
    from astropy.io import fits

    if fn == '':
        fn = './'
        pass
    if fn[-1] != '/':
        fn += '/'

    return fits.open(fn+catmap_filename)

def get_scuba2map(fn, catmap_filename="catnominal.FIT", prior="", **kwargs):
    '''
    return scuba2map class from directory name
    - prior: need to input if subset IQUmap to be registered. filename will be = {fn}/maps/{prior}_{I,Q,U}map.fits

    hints of kwargs:
    - imap: registered automatically in this function if not given
    - qmap: registered automatically in this function if not given
    - umap: registered automatically in this function if not given
    - iautomap: registered automatically in this function if not given
    - catmap: registered automatically in this function if not given
    - catinfo=True
    - delt_gal=0.0015
    - delt=None
    - moreinfo=False
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
            kwargs['imap'] = fits.open(fn+"iext.fits")
        pass
    if 'qmap' not in kwargs:
        if prior != "":
            kwargs['qmap'] = fits.open(fn+"/maps/"+prior+"_Qmap.fits")
        else:
            kwargs['qmap'] = fits.open(fn+"qext.fits")
        pass
    if 'umap' not in kwargs:
        if prior != "":
            kwargs['umap'] = fits.open(fn+"/maps/"+prior+"_Umap.fits")
        else:
            kwargs['umap'] = fits.open(fn+"uext.fits")
        pass
    if 'catmap' not in kwargs:
        try:
            kwargs['catmap'] = fits.open(fn+catmap_filename)
        except:
            print("ERROR:: Cannot register catmap.")
        pass
    if 'iautomap' not in kwargs:
        try:
            kwargs['iautomap'] = fits.open(fn+"iauto.fits")
        except:
            print("ERROR:: Cannot register iautomap.")
        pass

    return scuba2map(**kwargs)

class scuba2map():
    def __init__(self, imap, qmap, umap,
                 iautomap=None, catmap=None, catinfo=True,
                 delt_gal=0.0015, delt=None, moreinfo=False):

        if catmap is None:
            catinfo = False

        ihdr = imap[0].header
        qhdr = qmap[0].header
        uhdr = umap[0].header

        cdelt = ihdr['CDELT1']
        crval = ihdr['CRVAL1']
        crpix = ihdr['CRPIX1']
        naxis = np.min([ihdr['NAXIS1'], qhdr['NAXIS1'], uhdr['NAXIS1']])
        if iautomap is not None:
            naxis = np.min([naxis,iautomap[0].header['NAXIS1']])

        print(f'RA  axis: delta = {cdelt:+.4f}, center = {crval:+.4f} at index{crpix:.0f}, N = {naxis:.0f}')

        ra_axis = header2array(cdelt, crval, crpix, naxis)
        nra = naxis

        cdelt = ihdr['CDELT2']
        crval = ihdr['CRVAL2']
        crpix = ihdr['CRPIX2']
        naxis = np.min([ihdr['NAXIS2'], qhdr['NAXIS2'], uhdr['NAXIS2']])
        if iautomap is not None:
            naxis = np.min([naxis,iautomap[0].header['NAXIS2']])

        print(f'DEC axis: delta = {cdelt:+.4f}, center = {crval:+.4f} at index{crpix:.0f}, N = {naxis:.0f}')

        dec_axis = header2array(cdelt, crval, crpix, naxis)
        ndec = naxis

        if delt is None:
            rabins,dra   = array2bins(ra_axis,return_dx=True)
            decbins = array2bins(dec_axis,dra)
        else:
            dra = delt
            rabins  = array2bins(ra_axis,dra)
            decbins = array2bins(dec_axis,dra)

        print(f'[RA,DEC] in extmap: pixel size = {np.abs(dra):.2e} deg sq = {dra**2:.2e} deg2 --> {np.abs(dra)*3600:.2f} arcsec sq = {dra**2*3600*3600:.2f} arcsec2')

        ra = np.array([ra_axis.tolist()] * len(dec_axis)).flatten()
        dec = np.array([[idec]*len(ra_axis) for idec in dec_axis]).flatten()

        i = imap[0].data[0][:ndec, :nra].flatten()
        iauto  = None
        diauto = None
        if iautomap is not None:
            iauto  = iautomap[0].data[0][:ndec, :nra].flatten()
            diauto = np.sqrt(iautomap[1].data[0][:ndec, :nra].flatten())

        q = qmap[0].data[0][:ndec, :nra].flatten()
        u = umap[0].data[0][:ndec, :nra].flatten()

        di = np.sqrt(imap[1].data[0][:ndec, :nra].flatten())
        dq = np.sqrt(qmap[1].data[0][:ndec, :nra].flatten())
        du = np.sqrt(umap[1].data[0][:ndec, :nra].flatten())

        # convert galactic coord
        coord = SkyCoord(ra=ra*units.deg, dec=dec*units.deg, frame='fk5')
        gall = coord.galactic.l.deg
        galb = coord.galactic.b.deg

        gallbins = array2bins(gall, -1*delt_gal)
        galbbins = array2bins(galb, delt_gal)

        psi = parallactic_angle(ra, dec)

        gali, galq, galu = galactic_iqu(i, q, u, psi)
        galdi, galdq, galdu = galactic_iqu(di, dq, du, psi)

        self.i = i
        self.q = q
        self.u = u
        self.di = di
        self.dq = dq
        self.du = du

        self.iauto  = iauto
        self.diauto = diauto

        self.ra = ra
        self.dec = dec
        self.dpix = dra

        self.nra = nra
        self.ndec = ndec
        self.radecbins = (rabins, decbins)

        self.gi = gali
        self.gq = galq
        self.gu = galu
        self.gdi = galdi
        self.gdq = galdq
        self.gdu = galdu

        self.gl = gall
        self.gb = galb
        self.glgbbins = (gallbins, galbbins)
        self.psi = psi

        self.c_ra = None
        self.c_dec = None
        self.c_radecbins = (None, None)
        self.c_dpix = None

        self.c_gl = None
        self.c_gb = None
        self.c_glgbbins = (None, None)

        if catinfo:
            self._register_catinfo(cat=catmap,
                                   crpix1=ihdr['CRPIX1'],crpix2=ihdr['CRPIX2'],
                                   cdelt_gal=delt_gal, cdelt=delt, moreinfo=moreinfo)

        if moreinfo:
            self.header = ihdr
        return

    def _register_catinfo(self,cat,crpix1,crpix2,cdelt_gal,cdelt=None, moreinfo=False): #cdelt1,cdelt2,
        if cat is None:
            return

        def nanarray(dim):
            ret = np.zeros(dim)
            ret[:, :] = np.nan
            return ret

        dim = (self.ndec,self.nra)
        i = nanarray(dim)
        q = None
        u = None
        if moreinfo:
            q = nanarray(dim)
            u = nanarray(dim)
            di = nanarray(dim)
            dq = nanarray(dim)
            du = nanarray(dim)
            pass
        ra  = nanarray(dim)
        dec = nanarray(dim)
        am  = nanarray(dim)
        pm  = nanarray(dim)
        for x, (s, t) in enumerate(zip(cat[1].data.X, cat[1].data.Y)):
            i[int(t+crpix2-1)][int(s+crpix1-1)]  = cat[1].data.I[x]
            ra[int(t+crpix2-1)][int(s+crpix1-1)]  = cat[1].data.RA[x]*180./np.pi
            dec[int(t+crpix2-1)][int(s+crpix1-1)] = cat[1].data.DEC[x]*180./np.pi
            am[int(t+crpix2-1)][int(s+crpix1-1)]  = 0 if cat[1].data.AST[x]<-1 else 1
            pm[int(t+crpix2-1)][int(s+crpix1-1)]  = 0 if cat[1].data.PCA[x]<-1 else 1
            if moreinfo:
                q[int(t+crpix2-1)][int(s+crpix1-1)] = cat[1].data.Q[x]
                u[int(t+crpix2-1)][int(s+crpix1-1)] = cat[1].data.U[x]
                di[int(t+crpix2-1)][int(s+crpix1-1)]  = cat[1].data.DI[x]
                dq[int(t+crpix2-1)][int(s+crpix1-1)]  = cat[1].data.DQ[x]
                du[int(t+crpix2-1)][int(s+crpix1-1)]  = cat[1].data.DU[x]
                pass

        i = i.flatten()
        ra  = ra.flatten()
        dec = dec.flatten()
        am = am.flatten()
        pm = pm.flatten()
        if moreinfo:
            q = q.flatten()
            u = u.flatten()
            di = di.flatten()
            dq = dq.flatten()
            du = du.flatten()
            pass

        dra = cdelt
        if cdelt is None:
            dra  = np.median(np.diff(ra[~np.isnan(ra)]))
        rabins = array2bins(ra, dra)
        decbins = array2bins(dec, dra)

        print(f'[RA,DEC] in catmap: pixel size = {np.abs(dra):.2e} deg sq = {dra**2:.2e} deg2 --> {np.abs(dra)*3600:.2f} arcsec sq = {dra**2*3600*3600:.2f} arcsec2')

        # convert galactic coord
        coord = SkyCoord(ra=ra*units.deg, dec=dec*units.deg, frame='fk5')
        gall = coord.galactic.l.deg
        galb = coord.galactic.b.deg

        gallbins = array2bins(gall, -1*cdelt_gal)
        galbbins = array2bins(galb, cdelt_gal)

        self.c_ra = ra
        self.c_dec = dec
        self.c_radecbins = (rabins, decbins)
        self.c_dpix = dra

        self.c_gl = gall
        self.c_gb = galb
        self.c_glgbbins = (gallbins, galbbins)

        self.astmask = am
        self.pcamask = pm

        self.c_i = i
        if moreinfo:
            self.c_q = q
            self.c_u = u
            self.c_di = di
            self.c_dq = dq
            self.c_du = du
            psi = parallactic_angle(ra, dec)
            gali, galq, galu = galactic_iqu(i, q, u, psi)
            galdi, galdq, galdu = galactic_iqu(di, dq, du, psi)
            self.c_gi = gali
            self.c_gq = galq
            self.c_gu = galu
            self.c_gdi = galdi
            self.c_gdq = galdq
            self.c_gdu = galdu
            self.c_psi = psi
            pass

        return

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
    def glgb(self):
        return (self.gl, self.gb)

    @property
    def c_radec(self):
        return (self.c_ra, self.c_dec)

    @property
    def c_glgb(self):
        return (self.c_gl, self.c_gb)

    pass

