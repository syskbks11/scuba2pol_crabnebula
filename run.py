#!/usr/bin/env python3

SUBSETS = ['20210807_00067_0002',
           '20210807_00068_0002',
           '20210818_00067_0002',
           '20210825_00045_0002',
           '20210825_00046_0002']

def main(fdir, name_format, fn_astmask=None, fn_pcamask=None, do_subset=SUBSETS, mapsuffix=''):
    ## Graphic libraries
    import matplotlib.pyplot as plt

    ## Scientific libraries
    import numpy as np
    import pandas as pd

    from astropy.io import fits

    import scuba2tool
    import plotmap
    import polcalc

    # parameters

    # delt_gal = 6/3600
    # delt_fk5 = 5/3600

    # offsetx_rms = 0.1 #[deg]
    # offsety_rms = 0.09 #[deg]

    # r_signal = 0.09
    # r_rms    = 0.03

    if do_subset is None:
        do_subset = []
    if mapsuffix != '':
        mapsuffix = '_'+mapsuffix

    astmask = None
    pcamask = None
    if fn_astmask is not None:
        astmask = fits.open(fn_astmask)
    if fn_pcamask is not None:
        pcamask = fits.open(fn_pcamask)

    thismap = scuba2tool.get_scuba2map(fdir,astmask=astmask,pcamask=pcamask,ext=mapsuffix)
    thismap_sub = [scuba2tool.get_scuba2map(fdir,prior=p,astmask=astmask,pcamask=pcamask,wcs_target=thismap.wcs,data_target=thismap.i) for p in do_subset]

    cent = scuba2tool.get_tauAcent()
    fact = scuba2tool.get_fcf()

    # IQU
    fig,ax = plotmap.wcs_subplots(thismap.wcs, ncols=3)

    plotmap.plotmap(thismap.i*fact,thismap.wcs,title='I'+r'$_{\rm FK5}$'+' [mJy/arcsec2]', vmin=-0.2,vmax=4,fig=fig,ax=ax[0]) 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[0],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[0],transform='fk5',ls='-',facecolor='none',edgecolor='w')

    plotmap.plotmap(thismap.q*fact,thismap.wcs,title='Q'+r'$_{\rm FK5}$'+' [mJy/arcsec2]', vmin=-0.2,vmax=1,fig=fig,ax=ax[1])
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[1],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[1],transform='fk5',ls='-',facecolor='none',edgecolor='w')

    plotmap.plotmap(thismap.u*fact,thismap.wcs,title='U'+r'$_{\rm FK5}$'+' [mJy/arcsec2]', vmin=-1,vmax=0.2,fig=fig,ax=ax[2])
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[2],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[2],transform='fk5',ls='-',facecolor='none',edgecolor='w')

    fig.tight_layout()
    fig.savefig(name_format+"_iqu.pdf")
    plt.clf()
    plt.close()

    # Pol
    fig,ax = plotmap.wcs_subplots(thismap.wcs, ncols=3)

    d = polcalc.polamp(thismap.q,thismap.u) * fact
    s = np.abs(thismap.i/thismap.di) < 2
    d[s] = 0
    plotmap.plotmap(d,thismap.wcs,title='PolAmp'+r'$_{\rm FK5}$'+' [mJy/arcsec2]', vmin=0, vmax=1, fig=fig,ax=ax[0]) 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[0],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[0],transform='fk5',ls='-',facecolor='none',edgecolor='w')

    d = polcalc.polang(thismap.q,thismap.u)
    d[s] = 0
    plotmap.plotmap(d,thismap.wcs,title='PolAngle'+r'$_{\rm FK5}$'+' [deg]', vmin=30, vmax=180, fig=fig,ax=ax[1])
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[1],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[1],transform='fk5',ls='-',facecolor='none',edgecolor='w')

    d = polcalc.polfrac(thismap.i,thismap.q,thismap.u)
    d[s] = 0
    plotmap.plotmap(d,thismap.wcs,title='PolFrac'+r'$_{\rm FK5}$'+' ', vmin=0, vmax=0.3, fig=fig,ax=ax[2])
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[2],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[2],transform='fk5',ls='-',facecolor='none',edgecolor='w')

    fig.tight_layout()
    fig.savefig(name_format+"_pol.pdf")
    plt.clf()
    plt.close()

    # IQU GAL
    fig,ax = plotmap.wcs_subplots(thismap.gal.wcs, ncols=3)

    plotmap.plotmap(thismap.gal.i*fact,thismap.gal.wcs,title='I'+r'$_{\rm GAL}$'+' [mJy/arcsec2]',
            vmin=-0.2,vmax=4,fig=fig,ax=ax[0])
    plotmap.wcsaxes_circle((cent.galactic.l,cent.galactic.b),0.07,ax[0],
                              transform='galactic',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.galactic.l,cent.galactic.b),0.08,ax[0],
                              transform='galactic',ls='-',facecolor='none',edgecolor='w')

    plotmap.plotmap(thismap.gal.q*fact,thismap.gal.wcs,title='Q'+r'$_{\rm GAL}$'+' [mJy/arcsec2]',
            vmin=-1,vmax=0.2,fig=fig,ax=ax[1])
    plotmap.wcsaxes_circle((cent.galactic.l,cent.galactic.b),0.07,ax[1],
                              transform='galactic',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.galactic.l,cent.galactic.b),0.08,ax[1],
                              transform='galactic',ls='-',facecolor='none',edgecolor='w')

    plotmap.plotmap(thismap.gal.u*fact,thismap.gal.wcs,title='U'+r'$_{\rm GAL}$'+' [mJy/arcsec2]',
            vmin=-0.2,vmax=1,fig=fig,ax=ax[2])
    plotmap.wcsaxes_circle((cent.galactic.l,cent.galactic.b),0.07,ax[2],
                              transform='galactic',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.galactic.l,cent.galactic.b),0.08,ax[2],
                              transform='galactic',ls='-',facecolor='none',edgecolor='w')

    fig.tight_layout()
    fig.savefig(name_format+"_galiqu.pdf")
    plt.clf()
    plt.close()

    fig,ax = plotmap.wcs_subplots(thismap.gal.wcs, ncols=3)

    d = polcalc.polamp(thismap.gal.q,thismap.gal.u) * fact
    s = np.abs(thismap.gal.i/thismap.gal.di) < 2
    d[s] = 0
    plotmap.plotmap(d,thismap.gal.wcs,title='PolAmp'+r'$_{\rm GAL}$'+' [mJy/arcsec2]', vmin=0, vmax=1, fig=fig,ax=ax[0]) 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[0],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[0],transform='fk5',ls='-',facecolor='none',edgecolor='w')

    d = polcalc.polang(thismap.gal.q,thismap.gal.u)
    d[s] = 0
    plotmap.plotmap(d,thismap.gal.wcs,title='PolAngle'+r'$_{\rm GAL}$'+' [deg]', vmin=30, vmax=180, fig=fig,ax=ax[1])
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[1],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[1],transform='fk5',ls='-',facecolor='none',edgecolor='w')

    d = polcalc.polfrac(thismap.gal.i,thismap.gal.q,thismap.gal.u)
    d[s] = 0
    plotmap.plotmap(d,thismap.gal.wcs,title='PolFrac'+r'$_{\rm GAL}$'+' ', vmin=0, vmax=0.3, fig=fig,ax=ax[2])
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[2],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
    plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[2],transform='fk5',ls='-',facecolor='none',edgecolor='w')

    fig.tight_layout()
    fig.savefig(name_format+"_galpol.pdf")
    plt.clf()
    plt.close()

    # IQU,Pol in subobs

    if len(thismap_sub)>0:
        fig,ax = plotmap.wcs_subplots(thismap.wcs, ncols=6, nrows=len(thismap_sub))

        for i,asub in enumerate(thismap_sub):

            plotmap.plotmap(asub.i*fact,asub.wcs,title='I'+r'$_{\rm FK5}$'+' [mJy/arcsec2]'+f' (#{i})', vmin=-0.2,vmax=4,fig=fig,ax=ax[i*6+0])
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[i*6+0],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[i*6+0],transform='fk5',ls='-',facecolor='none',edgecolor='w')
            plotmap.plotmap(asub.q*fact,thismap.wcs,title='Q'+r'$_{\rm FK5}$'+' [mJy/arcsec2]'+f' (#{i})', vmin=-0.2,vmax=1,fig=fig,ax=ax[i*6+1])
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[i*6+1],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[i*6+1],transform='fk5',ls='-',facecolor='none',edgecolor='w')
            plotmap.plotmap(asub.u*fact,asub.wcs,title='U'+r'$_{\rm FK5}$'+' [mJy/arcsec2]'+f' (#{i})', vmin=-1,vmax=0.2,fig=fig,ax=ax[i*6+2])
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[i*6+2],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[i*6+2],transform='fk5',ls='-',facecolor='none',edgecolor='w')

            d = polcalc.polamp(asub.q,asub.u) * fact
            s = np.abs(asub.i/asub.di) < 2
            d[s] = 0
            plotmap.plotmap(d,asub.wcs,title='PolAmp'+r'$_{\rm FK5}$'+' [mJy/arcsec2]'+f' (#{i})', vmin=0, vmax=1, fig=fig,ax=ax[i*6+3]) 
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[i*6+3],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[i*6+3],transform='fk5',ls='-',facecolor='none',edgecolor='w')

            d = polcalc.polang(asub.q,asub.u)
            d[s] = 0
            plotmap.plotmap(d,asub.wcs,title='PolAngle'+r'$_{\rm FK5}$'+' [deg]'+f' (#{i})', vmin=30, vmax=180, fig=fig,ax=ax[i*6+4])
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[i*6+4],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[i*6+4],transform='fk5',ls='-',facecolor='none',edgecolor='w')

            d = polcalc.polfrac(asub.i,asub.q,asub.u)
            d[s] = 0
            plotmap.plotmap(d,asub.wcs,title='PolFrac'+r'$_{\rm FK5}$'+' '+f' (#{i})', vmin=0, vmax=0.3, fig=fig,ax=ax[i*6+5])
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.07,ax[i*6+5],transform='fk5',ls='-',facecolor='none',edgecolor='w') 
            plotmap.wcsaxes_circle((cent.ra,cent.dec),0.08,ax[i*6+5],transform='fk5',ls='-',facecolor='none',edgecolor='w')

            pass

        fig.tight_layout()
        fig.savefig(name_format+"_subobs.pdf")
        plt.clf()
        plt.close()


    # signal integration / aparture photometory

    cx = cent.ra.deg #[deg]
    cy = cent.dec.deg #[deg]
    x = thismap.ra #[deg]
    y = thismap.dec #[deg]
    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
    distf = dist.flatten()

    radius_list = 0.001*np.arange(1,100) #[deg]
    area_list = np.array([np.pi*ir**2*3600**2 for ir in radius_list]) #[arcsec2]
    #npix_list = np.array([len(np.where(distf<ir)[0]) for ir in radius_list])
    #area_list = 4*4*npix_list #[arcsec2]

    vals = thismap.i.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
    ret_i = np.array([np.mean(vals[distf<ir]) for ir in radius_list])*area_list # [Jy]

    vals = thismap.q.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
    ret_q = np.array([np.mean(vals[distf<ir]) for ir in radius_list])*area_list # [Jy]

    vals = thismap.u.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
    ret_u = np.array([np.mean(vals[distf<ir]) for ir in radius_list])*area_list # [Jy]

    ret_sub_i = [None]*len(thismap_sub)
    ret_sub_q = [None]*len(thismap_sub)
    ret_sub_u = [None]*len(thismap_sub)

    for i,asub in enumerate(thismap_sub):
        vals = asub.i.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
        ret_sub_i[i] = np.array([np.mean(vals[distf<ir]) for ir in radius_list])*area_list # [Jy]

        vals = asub.q.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
        ret_sub_q[i] = np.array([np.mean(vals[distf<ir]) for ir in radius_list])*area_list # [Jy]

        vals = asub.u.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
        ret_sub_u[i] = np.array([np.mean(vals[distf<ir]) for ir in radius_list])*area_list # [Jy]

        pass

    # aperture photometry
    r0 = 0.07
    r1 = 0.08
    ss = (distf>=r0) & (distf<r1)
    area = (np.pi*(r1**2-r0**2)*3600**2) # [arcsec2]

    fig,ax = plt.subplots(figsize=(18,5),ncols=3,sharey=True,sharex=True)
    ax = ax.flatten()

    vals = thismap.i.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
    noise_i = np.mean(vals[ss]) # [Jy/arcsec2]
    noiserms_i = np.std(vals[ss]) # [Jy/arcsec2]
    ax[0].hist(vals[ss]*1000,bins=np.arange(-0.4,0.4,0.01),color='r',alpha=0.7)

    vals = thismap.q.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
    noise_q = np.mean(vals[ss]) # [Jy/arcsec2]
    noiserms_q = np.std(vals[ss]) # [Jy/arcsec2]
    ax[1].hist(vals[ss]*1000,bins=np.arange(-0.4,0.4,0.01),color='b',alpha=0.7)

    vals = thismap.u.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
    noise_u = np.mean(vals[ss]) # [Jy/arcsec2]
    noiserms_u = np.std(vals[ss]) # [Jy/arcsec2]
    ax[2].hist(vals[ss]*1000,bins=np.arange(-0.4,0.4,0.01),color='k',alpha=0.7)

    ret_i_kai = ret_i - noise_i * area_list
    ret_q_kai = ret_q - noise_q * area_list
    ret_u_kai = ret_u - noise_u * area_list

    ax[0].set_title(f"noise I = {noise_i*1000:+9.6f} ± {noiserms_i*1000:5.2g} mJy/arcsec2")
    ax[1].set_title(f"noise Q = {noise_q*1000:+8.2g}  ± {noiserms_q*1000:5.2g} mJy/arcsec2")
    ax[2].set_title(f"noise U = {noise_u*1000:+8.2g}  ± {noiserms_u*1000:5.2g} mJy/arcsec2")

    for ia in ax:
        ia.set_xlabel("noise [mJy/arcsec2]")

    ax[0].set_ylabel("entries")

    fig.tight_layout()
    fig.savefig(name_format+"_noise.pdf")
    plt.clf()
    plt.close()

    ret_sub_i_kai = [None]*len(thismap_sub)
    ret_sub_q_kai = [None]*len(thismap_sub)
    ret_sub_u_kai = [None]*len(thismap_sub)

    noise_sub_i = [None]*len(thismap_sub)
    noise_sub_q = [None]*len(thismap_sub)
    noise_sub_u = [None]*len(thismap_sub)

    noiserms_sub_i = [None]*len(thismap_sub)
    noiserms_sub_q = [None]*len(thismap_sub)
    noiserms_sub_u = [None]*len(thismap_sub)

    if len(thismap_sub)>0:
        fig,ax = plt.subplots(figsize=(18,5*len(thismap_sub)),ncols=3,nrows=len(thismap_sub),sharey=True,sharex=True)
        ax = ax.flatten()

        for i,asub in enumerate(thismap_sub):
            vals = asub.i.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
            noise_sub_i[i] = np.mean(vals[ss]) # [Jy/arcsec2]
            noiserms_sub_i[i] = np.std(vals[ss]) # [Jy/arcsec2]
            ax[i*3+0].hist(vals[ss]*1000,bins=np.arange(-0.4,0.4,0.01),color='r',alpha=0.7)

            vals = asub.q.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
            noise_sub_q[i] = np.mean(vals[ss]) # [Jy/arcsec2]
            noiserms_sub_q[i] = np.std(vals[ss]) # [Jy/arcsec2]
            ax[i*3+1].hist(vals[ss]*1000,bins=np.arange(-0.4,0.4,0.01),color='b',alpha=0.7)

            vals = asub.u.flatten() * fact /1000 # [pW] --> [Jy/arcsec2]
            noise_sub_u[i] = np.mean(vals[ss]) # [Jy/arcsec2]
            noiserms_sub_u[i] = np.std(vals[ss]) # [Jy/arcsec2]
            ax[i*3+2].hist(vals[ss]*1000,bins=np.arange(-0.4,0.4,0.01),color='k',alpha=0.7)

            ret_sub_i_kai[i] = ret_sub_i[i] - noise_sub_i[i] * area_list
            ret_sub_q_kai[i] = ret_sub_q[i] - noise_sub_q[i] * area_list
            ret_sub_u_kai[i] = ret_sub_u[i] - noise_sub_u[i] * area_list

            ax[i*3+0].set_title(f"noise I = {noise_sub_i[i] *1000:+8.2g} ± {noiserms_sub_i[i] *1000:5.2g} mJy/arcsec2 (#{i})")
            ax[i*3+1].set_title(f"noise Q = {noise_sub_q[i] *1000:+8.2g}  ± {noiserms_sub_q[i] *1000:5.2g} mJy/arcsec2 (#{i})")
            ax[i*3+2].set_title(f"noise U = {noise_sub_u[i] *1000:+8.2g}  ± {noiserms_sub_u[i] *1000:5.2g} mJy/arcsec2 (#{i})")
            ax[i*3+0].set_ylabel("entries")

        ax[(len(thismap_sub)-1)*3+0].set_xlabel("noise [mJy/arcsec2]")
        ax[(len(thismap_sub)-1)*3+1].set_xlabel("noise [mJy/arcsec2]")
        ax[(len(thismap_sub)-1)*3+2].set_xlabel("noise [mJy/arcsec2]")

        fig.tight_layout()
        fig.savefig(name_format+"_noise_subobs.pdf")
        plt.clf()
        plt.close()

    fig,ax = plt.subplots(figsize=(18,8),ncols=3,nrows=2)
    ax = ax.flatten()

    ax[0].plot(radius_list*3600,ret_i, 'o:',c='orange',alpha=0.5)
    ax[1].plot(radius_list*3600,ret_q, 'o:',c='orange',alpha=0.5)
    ax[2].plot(radius_list*3600,ret_u, 'o:',c='orange',alpha=0.5)
    ax[3].plot(radius_list*3600,polcalc.polamp(ret_q,ret_u), 'o:',c='orange',alpha=0.5)
    ax[4].plot(radius_list*3600,polcalc.polang(ret_q,ret_u), 'o:',c='orange',alpha=0.5)
    ax[5].plot(radius_list*3600,polcalc.polfrac(ret_i,ret_q,ret_u), 'o:',c='orange',alpha=0.5)

    ax[0].plot(radius_list*3600,ret_i_kai, 'bo:')
    ax[0].set_ylabel("Integrated I [Jy]")
    ax[1].plot(radius_list*3600,ret_q_kai, 'bo:')
    ax[1].set_ylabel("Integrated Q [Jy]")
    ax[2].plot(radius_list*3600,ret_u_kai, 'bo:')
    ax[2].set_ylabel("Integrated U [Jy]")
    ax[3].plot(radius_list*3600,polcalc.polamp(ret_q_kai,ret_u_kai), 'bo:')
    ax[3].set_ylabel("Integrated Ampl"+r"$_{\rm pol}$ [Jy]")
    ax[4].plot(radius_list*3600,polcalc.polang(ret_q_kai,ret_u_kai), 'bo:')
    ax[4].set_ylabel("Integrated Angle"+r"$_{\rm pol}$ [deg]")
    ax[5].plot(radius_list*3600,polcalc.polfrac(ret_i_kai,ret_q_kai,ret_u_kai), 'bo:')
    ax[5].set_ylabel("Integrated Frac"+r"$_{\rm pol}$")

    for iax in ax:
        iax.set_xlabel("Integrated radius [arcsec]")

    fig.tight_layout()
    fig.savefig(name_format+"_integrated.pdf")
    plt.clf()
    plt.close()

    fig,ax = plt.subplots(figsize=(18,8),ncols=3,nrows=2, sharex=True)
    ax = ax.flatten()

    for i,asub in enumerate(thismap_sub):
        ax[0].plot(radius_list*3600,ret_sub_i_kai[i], 'o:',label=f'#{i}')
        ax[1].plot(radius_list*3600,ret_sub_q_kai[i], 'o:',label=f'#{i}')
        ax[2].plot(radius_list*3600,ret_sub_u_kai[i], 'o:',label=f'#{i}')
        ax[3].plot(radius_list*3600,polcalc.polamp(ret_sub_q_kai[i],ret_sub_u_kai[i]), 'o:',label=f'#{i}')
        ax[4].plot(radius_list*3600,polcalc.polang(ret_sub_q_kai[i],ret_sub_u_kai[i]), 'o:',label=f'#{i}')
        ax[5].plot(radius_list*3600,polcalc.polfrac(ret_sub_i_kai[i],ret_sub_q_kai[i],ret_sub_u_kai[i]), 'o:',label=f'#{i}')


    ax[0].plot(radius_list*3600,ret_i_kai, 'ko:',label='all')
    ax[0].set_ylabel("Integrated I [Jy]")
    ax[1].plot(radius_list*3600,ret_q_kai, 'ko:',label='all')
    ax[1].set_ylabel("Integrated Q [Jy]")
    ax[2].plot(radius_list*3600,ret_u_kai, 'ko:',label='all')
    ax[2].set_ylabel("Integrated U [Jy]")
    ax[3].plot(radius_list*3600,polcalc.polamp(ret_q_kai,ret_u_kai), 'ko:',label='all')
    ax[3].set_ylabel("Integrated Ampl"+r"$_{\rm pol}$ [Jy]")
    ax[4].plot(radius_list*3600,polcalc.polang(ret_q_kai,ret_u_kai), 'ko:',label='all')
    ax[4].set_ylabel("Integrated Angle"+r"$_{\rm pol}$ [deg]")
    ax[5].plot(radius_list*3600,polcalc.polfrac(ret_i_kai,ret_q_kai,ret_u_kai), 'ko:',label='all')
    ax[5].set_ylabel("Integrated Frac"+r"$_{\rm pol}$")

    for iax in ax[3:6]:
        iax.set_xlabel("Integrated radius [arcsec]")

    for iax in ax:
        iax.legend()

    fig.tight_layout()
    fig.savefig(name_format+"_integrated_subobs.pdf")
    plt.clf()
    plt.close()


    fig,ax = plotmap.wcs_subplots(thismap.wcs,ncols=4)
    if thismap.astmask is not None: 
        v = thismap.i.copy()*fact
        v[~np.isnan(thismap.astmask)] = np.nan
        plotmap.plotmap(v, thismap.wcs, fig=fig,ax=ax[0], vmin=-0.2, vmax=0.2, title='ASTmasked intensity '+r'[Jy/$\rm arcsec^2$]')
        v = thismap.i.copy()*fact
        v[np.isnan(thismap.astmask)] = np.nan
        plotmap.plotmap(v,thismap.wcs, fig=fig,ax=ax[1], vmin=-0.2, vmax=0.2, title='ASTmasked intensity '+r'[Jy/$\rm arcsec^2$]')
    if thismap.pcamask is not None:
        v = thismap.i.copy()*fact
        v[~np.isnan(thismap.pcamask)] = np.nan
        plotmap.plotmap(v, thismap.wcs, fig=fig,ax=ax[2], vmin=-0.2, vmax=0.2, title='PCAmasked intensity '+r'[Jy/$\rm arcsec^2$]')
        v = thismap.i.copy()*fact
        v[np.isnan(thismap.pcamask)] = np.nan
        plotmap.plotmap(v,thismap.wcs, fig=fig,ax=ax[3], vmin=-0.2, vmax=0.2, title='PCAmasked intensity '+r'[Jy/$\rm arcsec^2$]')
    fig.tight_layout()
    fig.savefig(name_format+"_mask.pdf")
    plt.clf()
    plt.close()

    ff = open(name_format+'_latex.tex', 'w')
    ff = open(name_format+'_latex.tex', 'a')

    colnames = ['$I$ [Jy]','$Q$ [Jy]','$U$ [Jy]','$I_p$ [Jy]','$\psi_p$ [deg]', '$p$']
    df = pd.DataFrame(index=[],columns=colnames)

    ind = np.argmin(np.abs(radius_list-0.08))

    pamp = polcalc.polamp(ret_q_kai,ret_u_kai)
    pang = polcalc.polang(ret_q_kai,ret_u_kai)
    pfrc = polcalc.polfrac(ret_i_kai,ret_q_kai,ret_u_kai)
    v = [ret_i_kai[ind],ret_q_kai[ind],ret_u_kai[ind],
         pamp[ind], pang[ind], pfrc[ind]]

    record = pd.Series(v, index=df.columns, name='all')
    df = pd.concat([df,record.to_frame().T])

    for i,asub in enumerate(thismap_sub):
        pamp = polcalc.polamp(ret_sub_q_kai[i],ret_sub_u_kai[i])
        pang = polcalc.polang(ret_sub_q_kai[i],ret_sub_u_kai[i])
        pfrc = polcalc.polfrac(ret_sub_i_kai[i],ret_sub_q_kai[i],ret_sub_u_kai[i])
        v = [ret_sub_i_kai[i][ind],ret_sub_q_kai[i][ind],ret_sub_u_kai[i][ind],
             pamp[ind], pang[ind], pfrc[ind]]

        record = pd.Series(v, index=df.columns, name=f'#{i}')
        df = pd.concat([df,record.to_frame().T])

    print("\\begin{tabular}{crrrrrrr}",file=ff)
    print("\\toprule",file=ff)
    print(' ', end=' & ',file=ff)
    for x in df.columns.values:
        if x == colnames[-1]:
            estr = ""
        else:
            estr = " & "
        print(f'{x}', end=estr,file=ff)
    print(" \\\\",file=ff)
    print("\\midrule",file=ff)
    for x,y in df.T.items():
        x = x.replace("#","\\#")
        print(f'{x}', end=" & ",file=ff)
        for k,v in y.items():
            if k == colnames[-1]:
                estr = ""
            else:
                estr = " & "
            print(f'{v:.4g}', end=estr,file=ff)
        print(" \\\\",file=ff)
    print("\\bottomrule",file=ff)
    print("\\end{tabular}",file=ff)
    print("\\label{tab:"+name_format+"_integ}",file=ff)

    print('',file=ff)
    print('',file=ff)
    print('',file=ff)

    colnames = ['$I_{\\rm noise}$ [mJy/$\\rm arcsec^2$]','$Q_{\\rm noise}$ [mJy/$\\rm arcsec^2$]','$U_{\\rm noise}$ [mJy/$\\rm arcsec^2$]']
    df = pd.DataFrame(index=[],columns=colnames)

    v = [f'{noise_i*1000:+8.2g} $\\pm$ {noiserms_i*1000:5.2g}',
         f'{noise_q*1000:+8.2g} $\\pm$ {noiserms_q*1000:5.2g}',
         f'{noise_u*1000:+8.2g} $\\pm$ {noiserms_u*1000:5.2g}']

    record = pd.Series(v, index=df.columns, name='all')
    df = pd.concat([df,record.to_frame().T])

    for i,asub in enumerate(thismap_sub):
        v = [f'{noise_sub_i[i]*1000:+8.2g} $\\pm$ {noiserms_sub_i[i]*1000:5.2g}',
             f'{noise_sub_q[i]*1000:+8.2g} $\\pm$ {noiserms_sub_q[i]*1000:5.2g}',
             f'{noise_sub_u[i]*1000:+8.2g} $\\pm$ {noiserms_sub_u[i]*1000:5.2g}']


        record = pd.Series(v, index=df.columns, name=f'#{i}')
        df = pd.concat([df,record.to_frame().T])

    print("\\begin{tabular}{crrr}",file=ff)
    print("\\toprule",file=ff)
    print(' ', end=' & ',file=ff)
    for x in df.columns.values:
        if x == colnames[-1]:
            estr = ""
        else:
            estr = " & "
        print(f'{x}', end=estr,file=ff)
    print(" \\\\",file=ff)
    print("\\midrule",file=ff)
    for x,y in df.T.items():
        x = x.replace("#","\\#")
        print(f'{x}', end=" & ",file=ff)
        for k,v in y.items():
            if k == colnames[-1]:
                estr = ""
            else:
                estr = " & "
            print(f'{v}', end=estr,file=ff)
        print(" \\\\",file=ff)
    print("\\bottomrule",file=ff)
    print("\\end{tabular}",file=ff)
    print("\\label{tab:"+name_format+"_noise}",file=ff)


    print('',file=ff)
    print('',file=ff)
    print('',file=ff)

    fignames = [
        name_format+"_iqu.pdf",
        name_format+"_pol.pdf",
        name_format+"_galiqu.pdf",
        name_format+"_galpol.pdf",
        name_format+"_subobs.pdf",
        name_format+"_noise.pdf",
        name_format+"_noise_subobs.pdf",
        name_format+"_integrated.pdf",
        name_format+"_integrated_subobs.pdf",
        name_format+"_astmask.pdf",
    ]

    for x in fignames:
        print("\\begin{figure}",file=ff)
        print("\\centering",file=ff)
        print("\\includegraphics[width=0.9\\textwidth]{"+x+"}",file=ff)
        print("\\caption{}",file=ff)
        print("\\label{fig:"+x.split("/")[-1].split(".")[0]+"}",file=ff)
        print("\\end{figure}",file=ff)
        print('',file=ff)


if __name__ == '__main__':

    ## System libraries
    import os
    import platform
    print(platform.node(), platform.platform())
    import sys
    print(sys.version, sys.platform, sys.executable)

    import argparse
    parser = argparse.ArgumentParser(description='main script for processing output of starlink pol2map. required files:\n (iext.fits,qext.fits,uext.fits)')
    parser.add_argument('dirname', help='directory to be processed')
    parser.add_argument('--prefix', default='', help='prefix for output files (file names will be "[prefix]/[prefix]_*")')
    parser.add_argument('--astmask', default=None, help='astmask')
    parser.add_argument('--pcamask', default=None, help='pcamask')
    parser.add_argument('--mapsuffix', default='', help='suffix for input map (map names will be "{i,q,u}ext_[mapsuffix].fits")')

    args = parser.parse_args()

    fdir = args.dirname
    prefix = fdir.split("/")[-1]
    if args.prefix != '':
        prefix = args.prefix
        pass

    os.makedirs(prefix,exist_ok=True)
    fileformat = prefix+'/'+prefix

    if args.astmask is None:
        if os.path.isfile(fdir+'/'+'astmask.fits'):
            args.astmask = fdir+'/'+'astmask.fits'
            pass
        pass
    if args.pcamask is None:
        if os.path.isfile(fdir+'/'+'pcamask.fits'):
            args.pcamask = fdir+'/'+'pcamask.fits'
            pass
        pass

    main(fdir,fileformat,fn_astmask=args.astmask,fn_pcamask=args.pcamask,mapsuffix=args.mapsuffix)
    #main(fdir,name_format2, do_subset=None, mapsuffix=name_format)
    #,fn_astmask='tauA/star21_850um_customPca/mask_nika.fits', fn_pcamask='tauA/star21_850um_customPca/mask_nika.fits')

