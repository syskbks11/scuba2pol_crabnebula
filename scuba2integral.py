#!python3
import matplotlib.pyplot as plt

import numpy as np

import scuba2tool

def _plot_comp_OLD(x,i,q,u,pamp,pang,pfrc,n,fig=None,ax=None,c='b',units=None,label=None,xlabel="",ylabel=""):
    if fig is None:
        fig, ax = plt.subplots(ncols=7, figsize=(28, 4))
        pass
    ax[0].plot(x, i,    'o', color=c, label=label)
    ax[1].plot(x, q,    'o', color=c, label=label)
    ax[2].plot(x, u,    'o', color=c, label=label)
    ax[3].plot(x, pamp, 'o', color=c, label=label)
    ax[4].plot(x, pang, 'o', color=c, label=label)
    ax[5].plot(x, pfrc, 'o', color=c, label=label)
    ax[6].plot(x, n, 'o', color=c, label=label)

    for ii,iax in enumerate(ax):
        if label is not None:
            iax.legend()
        iax.set_xlabel(xlabel)
        if units is not None:
            iax.set_ylabel(ylabel+" "+units[ii])
        else:
            iax.set_ylabel(ylabel)

    ax[5].set_ylabel('frac')
    ax[6].set_ylabel('# of pixels')

    ax[0].set_title("I")
    ax[1].set_title("Q")
    ax[2].set_title("U")
    ax[3].set_title("PolAmpl")
    ax[4].set_title("PolAngle")
    ax[5].set_title("PolFrac")
    ax[6].set_title("# of Pixels")
    fig.tight_layout()

    return fig,ax

def _plot_comp(x,ys,fig=None,ax=None,c='b',label=None,xlabel="",ylabels=None,titles=None):
    if fig is None:
        fig, ax = plt.subplots(ncols=len(ys), figsize=(4*len(ys), 4))
        pass

    for iax,iy in zip(ax,ys):
        iax.plot(x, iy, 'o', color=c, label=label)

    for ii,iax in enumerate(ax):
        if label is not None:
            iax.legend()

        iax.set_xlabel(xlabel)

        if ylabels is not None:
            iax.set_ylabel(ylabels[ii])

        if titles is not None:
            iax.set_title(titles[ii])

    fig.tight_layout()

    return fig,ax

def calc_rms(i, q, u, coord, cent, dist0, dist1=None,
             fcf=495*1.35*1, fig=None,ax=None, c='b', un="", do_print=True, label=None):

    if dist1 is None:
        dist1 = dist0*2
    dist = np.sqrt((coord[0]-cent[0])**2 + (coord[1]-cent[1])**2)
    distx = np.arange(1e-3, dist1, 1e-4)

    pamp = scuba2tool.polamp(q, u)
    pang = scuba2tool.polang(q, u)
    pfrc = scuba2tool.polfrac(i, q, u)

    nint = np.zeros_like(distx)
    iint = np.zeros_like(distx)
    qint = np.zeros_like(distx)
    uint = np.zeros_like(distx)
    pfrcint = np.zeros_like(distx)
    pampint = np.zeros_like(distx)
    pangint = np.zeros_like(distx)
    for ii, ix in enumerate(distx):
        ss = dist < ix
        nint[ii] = len(i[ss]) - np.sum(np.isnan(i[ss]))
        iint[ii] = np.nanstd(i[ss])*fcf
        qint[ii] = np.nanstd(q[ss])*fcf
        uint[ii] = np.nanstd(u[ss])*fcf
        pfrcint[ii] = np.nanstd(pfrc[ss])
        pampint[ii] = np.nanstd(pamp[ss])*fcf
        pangint[ii] = np.nanstd(pang[ss])

    ylabels = ["rms "+un, "rms "+un, "rms "+un,
               "rms "+un, "rms [deg]", "rms", "# of pixels"]
    _plot_comp(x=distx*3600,ys=[iint,qint,uint,pampint,pangint,pfrcint,nint],
               fig=fig,ax=ax,c=c,label=label,
               xlabel=f'from ({cent[0]:.1f},{cent[1]:.1f}) [arcsec]',
               ylabels=ylabels,
               titles=["I", "Q", "U", "PolAmpl", "PolAngle", "PolFrac", "# of pixels"])

    x = np.argmin(np.abs(distx-dist0))
    if do_print:
        print(f'I     = {iint[x] }')
        print(f'Q     = {qint[x] }')
        print(f'U     = {uint[x] }')
        print(f'PFRAC = {pfrcint[x]}')
        print(f'ANG   = {pangint[x]}')
        print(f'NPIX  = {nint[x]}')
        pass

    return [iint[x],qint[x],uint[x],pampint[x],pangint[x],pfrcint[x],int(nint[x]),dist0*3600]

def calc_mean(i, q, u, coord, cent, dist0, dist1=None,
              fcf=495*1.35*1, fig=None,ax=None, c='b', un="", do_print=True, label=None):

    if dist1 is None:
        dist1 = dist0*2

    dist = np.sqrt((coord[0]-cent[0])**2 + (coord[1]-cent[1])**2)
    distx = np.arange(1e-3, dist1, 1e-4)

    pamp = scuba2tool.polamp(q, u)
    pang = scuba2tool.polang(q, u)
    pfrc = scuba2tool.polfrac(i, q, u)

    nint = np.zeros_like(distx)
    iint = np.zeros_like(distx)
    qint = np.zeros_like(distx)
    uint = np.zeros_like(distx)
    pfrcint = np.zeros_like(distx)
    pampint = np.zeros_like(distx)
    pangint = np.zeros_like(distx)
    for ii, ix in enumerate(distx):
        ss = (dist < ix) # & (tmp==0)
        nint[ii] = len(i[ss]) - np.sum(np.isnan(i[ss]))
        iint[ii] = np.nanmean(i[ss])*fcf
        qint[ii] = np.nanmean(q[ss])*fcf
        uint[ii] = np.nanmean(u[ss])*fcf
        pfrcint[ii] = np.nanmean(pfrc[ss])
        pampint[ii] = np.nanmean(pamp[ss])*fcf
        pangint[ii] = np.nanmean(pang[ss])

    ylabels = ["mean "+un, "mean "+un, "mean "+un,
               "mean "+un, "mean [deg]", "mean", "# of pixels"]
    _plot_comp(x=distx*3600,ys=[iint,qint,uint,pampint,pangint,pfrcint,nint],
               fig=fig,ax=ax,c=c,label=label,
               xlabel=f'from ({cent[0]:.1f},{cent[1]:.1f}) [arcsec]',
               ylabels=ylabels,
               titles=["I", "Q", "U", "PolAmpl", "PolAngle", "PolFrac", "# of pixels"])

    x = np.argmin(np.abs(distx-dist0))
    if do_print:
        print(f'I     = {iint[x] }')
        print(f'Q     = {qint[x] }')
        print(f'U     = {uint[x] }')
        print(f'PFRAC = {pfrcint[x]}')
        print(f'ANG   = {pangint[x]}')
        print(f'NPIX  = {nint[x]}')
        pass
    return [iint[x],qint[x],uint[x],pampint[x],pangint[x],pfrcint[x],int(nint[x]),dist0*3600]

def calc_integration(i, q, u, di, dq, du, coord, cent, dist0, dist1=None,
                     fcf=495*1.35*1e3, fig=None,ax=None, c='b', un="", do_print=True, label=None):

    dist = np.sqrt((coord[0]-cent[0])**2 + (coord[1]-cent[1])**2)
    if dist1 is None:
        dist1 = dist0*1.5
    distx = np.arange(1e-3, dist1, 2e-3)

    nint = np.zeros_like(distx)
    iint = np.zeros_like(distx)
    qint = np.zeros_like(distx)
    uint = np.zeros_like(distx)
    #d = dq*du*di/np.sqrt((di*dq)**2+(dq*du)**2+(du*di)**2)
    #d = d/np.nanmean(d)
    for ii, ix in enumerate(distx):
        ss = dist < ix
        nint[ii] = len(i[ss]) - np.sum(np.isnan(i[ss]))
        iint[ii] = np.nansum(i[ss])*fcf
        qint[ii] = np.nansum(q[ss])*fcf
        uint[ii] = np.nansum(u[ss])*fcf

    pfint   = scuba2tool.polfrac(iint, qint, uint)
    pangint = scuba2tool.polang(qint, uint)
    paint   = scuba2tool.polamp(qint, uint)

    ylabels = ["integrated "+un, "integrated "+un, "integrated "+un,
               "integrated "+un, "integrated [deg]", "integrated", "# of pixels"]
    _plot_comp(x=distx*3600,ys=[iint,qint,uint,paint,pangint,pfint,nint],
               fig=fig,ax=ax,c=c,label=label,
               xlabel=f'from ({cent[0]:.1f},{cent[1]:.1f}) [arcsec]',
               ylabels=ylabels,
               titles=["I", "Q", "U", "PolAmpl", "PolAngle", "PolFrac", "# of pixels"])

    # print(distx,target)
    x = np.argmin(np.abs(distx-dist0))
    if do_print:
        print(f'I     = {iint[x] }')
        print(f'Q     = {qint[x] }')
        print(f'U     = {uint[x] }')
        print(f'PFRAC = {pfint[x]}')
        print(f'ANG   = {pangint[x]}')
        print(f'NPIX  = {nint[x]}')
        pass
    return [iint[x],qint[x],uint[x],paint[x],pangint[x],pfint[x],int(nint[x]),dist0*3600]

def calc_noise(i, q, u, coord, cent, dist0, dist1=None,
               fcf=495*1.35*1, fig=None,ax=None, c='b', un="", do_print=True, label=None):

    if dist1 is None:
        dist1 = dist0*2
    dist = np.sqrt((coord[0]-cent[0])**2 + (coord[1]-cent[1])**2)
    distx = np.arange(1e-3, dist1, 1e-4)

    nint = np.zeros_like(distx)
    iint = np.zeros_like(distx)
    qint = np.zeros_like(distx)
    uint = np.zeros_like(distx)
    pfrcint = np.zeros_like(distx)
    pampint = np.zeros_like(distx)
    pangint = np.zeros_like(distx)
    diint = np.zeros_like(distx)
    dqint = np.zeros_like(distx)
    duint = np.zeros_like(distx)
    dpfrcint = np.zeros_like(distx)
    dpampint = np.zeros_like(distx)
    dpangint = np.zeros_like(distx)
    #d = dq*du*di/np.sqrt((di*dq)**2+(dq*du)**2+(du*di)**2)
    for ii, ix in enumerate(distx):
        ss = dist < ix
        nint[ii] = len(i[ss]) - np.sum(np.isnan(i[ss]))
        iint[ii] = np.nanmean(i[ss])*fcf
        qint[ii] = np.nanmean(q[ss])*fcf
        uint[ii] = np.nanmean(u[ss])*fcf

        diint[ii] = np.nanstd(i[ss])*fcf
        dqint[ii] = np.nanstd(q[ss])*fcf
        duint[ii] = np.nanstd(u[ss])*fcf

    pfrcint = scuba2tool.polfrac(iint, qint, uint)
    pampint = scuba2tool.polamp(qint, uint)
    pangint = scuba2tool.polang(qint, uint)

    dpfrcint = scuba2tool.dpolfrac(iint, qint, uint, diint, dqint, duint, simple=True)
    dpampint = scuba2tool.dpolamp(qint, uint, dqint, duint, simple=True)
    dpangint = scuba2tool.dpolang(qint, uint, dqint, duint, simple=True)

    ylabels = ["deviation "+un, "deviation "+un, "deviation "+un, "deviation "+un, "# of pixels"]
    _plot_comp(x=distx*3600,ys=[diint,dqint,duint,dpampint,nint],
               fig=fig,ax=ax,c=c,label=label,
               #xlabel=f'from ({cent[0]:.1f},{cent[1]:.1f}) [arcsec]',
               xlabel='radius [arcsec]',
               ylabels=ylabels,
               titles=["I", "Q", "U", "PolAmpl", "# of pixels"])

    x = np.argmin(np.abs(distx-dist0))
    if do_print:
        print(f'I     = {iint[x] }')
        print(f'Q     = {qint[x] }')
        print(f'U     = {uint[x] }')
        print(f'PFRAC = {pfrcint[x]}')
        print(f'ANG   = {pangint[x]}')
        print(f'NPIX  = {nint[x]}')
        pass

    return [diint[x],dqint[x],duint[x],dpampint[x],int(nint[x]),dist0*3600]
