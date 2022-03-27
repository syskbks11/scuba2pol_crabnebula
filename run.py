#!/usr/bin/env python3
# coding: utf-8

# In[1]:


## Graphic libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

## Scientific libraries
import numpy as np
import pandas as pd
import scipy
# from scipy import signal
# from scipy.signal import savgol_filter

## Other libraries
import glob
import time
import datetime

## System libraries
import os
import platform
print(platform.node(), platform.platform())
import sys
print(sys.version, sys.platform, sys.executable)

## Setting matplotlib
mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'axes.facecolor':'w'})
mpl.rcParams.update({'axes.edgecolor':'k'})
mpl.rcParams.update({'figure.facecolor':'w'})
mpl.rcParams.update({'figure.edgecolor':'w'})
mpl.rcParams.update({'axes.grid':True})
mpl.rcParams.update({'grid.linestyle':':'})
mpl.rcParams.update({'figure.figsize':[12,9]})

# In[2]:


import scuba2tool, scuba2integral
from scuba2tool import plotmap


# In[3]:

def main(fdir,name_format, do_subset=True):

    # parameters
    delt_gal = 6/3600
    delt_fk5 = 5/3600

    offsetx_rms = 0.1 #[deg]
    offsety_rms = 0.09 #[deg]

    r_signal = 0.09
    r_rms    = 0.03

    # In[4]:

    catmap = scuba2tool.get_catmap(fdir)

    a = scuba2tool.get_scuba2map(fdir,catinfo=True,delt_gal=delt_gal,delt=delt_fk5,catmap=catmap)

    subset_priors = ['20210807_00067_0002',
                     '20210807_00068_0002',
                     '20210818_00067_0002',
                     '20210825_00045_0002',
                     '20210825_00046_0002']

    asub = []
    if do_subset:
        asub = [scuba2tool.get_scuba2map(fdir,prior=p,catinfo=True,delt_gal=delt_gal,delt=delt_fk5,catmap=catmap) for p in subset_priors]

    # In[5]:

    fcf_beam = scuba2tool.get_fcfbeam()
    fcf = scuba2tool.get_fcf()

    cent = scuba2tool.get_tauAcent()

    cra = cent.ra.deg
    cdec = cent.dec.deg

    radeccent = (cra, cdec)
    radeccent_l = (cra-offsetx_rms, cdec)
    radeccent_r = (cra+offsetx_rms, cdec)
    radeccent_u = (cra, cdec+offsety_rms)
    radeccent_b = (cra, cdec-offsety_rms)

    cgl = cent.galactic.l.deg
    cgb = cent.galactic.b.deg

    glgbcent  = (cgl, cgb)


    # In[6]:

    fig, ax = plt.subplots(figsize=(21, 4), ncols=4)

    ymax = np.max([np.nanmax(a.dec),np.nanmax(a.c_dec)]) +0.01
    ymin = np.min([np.nanmin(a.dec),np.nanmin(a.c_dec)]) -0.01
    xmax = np.max([np.nanmax(a.ra), np.nanmax(a.c_ra)]) +0.01
    xmin = np.min([np.nanmin(a.ra), np.nanmin(a.c_ra)]) -0.01

    #vmin=np.nanmin(a.i*495*1.35*1e3)*0.2
    #vmax=np.nanmax(a.i*495*1.35*1e3)*0.2

    vmin=-160
    vmax=220

    plotmap(a.radec, a.i*fcf_beam, a.radecbins, fig=fig, ax=ax[0],vmin=vmin,vmax=vmax)
    ax[0].set_title("Iext [mJy/beam]")
    ax[0].set_xlim(xmin,xmax)
    ax[0].set_ylim(ymin,ymax)

    plotmap(a.radec, a.iauto*fcf_beam, a.radecbins, fig=fig, ax=ax[1],vmin=vmin,vmax=vmax)
    ax[1].set_title("Iauto [mJy/beam]")
    ax[1].set_xlim(xmin,xmax)
    ax[1].set_ylim(ymin,ymax)

    plotmap(a.radec, (a.iauto - a.i)*fcf_beam, a.radecbins, fig=fig, ax=ax[2],vmin=vmin*0.1,vmax=vmax*0.1)
    ax[2].set_title("Iauto - Iext [mJy/beam]")
    ax[2].set_xlim(xmin,xmax)
    ax[2].set_ylim(ymin,ymax)

    plotmap(a.c_radec, a.c_i, a.c_radecbins, fig=fig, ax=ax[3],vmin=vmin,vmax=vmax)
    ax[3].set_title("Icat [mJy/beam]")
    ax[3].set_xlim(xmin,xmax)
    ax[3].set_ylim(ymin,ymax)

    fig.tight_layout()
    plt.clf()
    plt.close()

    #Comparison with cat

    fig, ax = plt.subplots(figsize=(21, 4), ncols=4)
    ax[0].plot(a.radec[0], a.radec[0]-a.c_radec[0], 'b.')
    ax[1].plot(a.radec[1], a.radec[1]-a.c_radec[1], 'r.')
    ax[0].set_title("Diff. btw simple RA and RA in catfile")
    ax[1].set_title("Diff. btw simple DEC and DEC in catfile")
    ax[0].set_ylabel(r"$\Delta$ RA [deg]")
    ax[1].set_ylabel(r"$\Delta$ DEC [deg]")
    ax[0].set_xlabel("RA (simple) [deg]")
    ax[1].set_xlabel("DEC (simple) [deg]")

    plotmap(a.radec, a.c_i/(a.i*fcf_beam), a.radecbins, fig=fig, ax=ax[2], vmin=1-3e-8, vmax=1+3e-8)
    ax[2].set_title("Icat/Iext")
    ax[2].set_xlim(xmin,xmax)
    ax[2].set_ylim(ymin,ymax)

    ax[3].hist(a.c_i/(a.i*fcf_beam),bins=100)
    ax[3].set_ylabel('number')
    ax[3].set_xlabel('Icat/Iext')
    ax[3].set_ylim(None,850)

    fig.tight_layout()
    # Not sure for a reason of these differences...


    # In[7]:


    fig, ax = plt.subplots(figsize=(12, 6))
    plt.hist(a.psi, bins=200, color='b', alpha=0.7)
    plt.ylabel("#")
    plt.xlabel("parallactic angle [deg]")
    plt.clf()
    plt.close()


    # In[8]:


    fig, ax = plt.subplots(figsize=(8, 6))
    #tmp = np.copy(a.astmask)
    #tmp[np.isnan(a.astmask)&(~np.isnan(autoi.i))] = 0
    plotmap(a.c_radec, np.abs(a.iauto/a.diauto), a.c_radecbins,
            fig=fig, ax=ax, count=5, vmin=0, vmax=5)
    plt.title(r'I$\rm_{FK5}$ S/N')

    plt.clf()
    plt.close()


    # In[9]:


    keys = [a.i*fcf, a.q*fcf, a.u*fcf]
    name = [r'I$\rm_{FK5}$ [mJy/arcsec$^2$]', r'Q$\rm_{FK5}$ [mJy/arcsec$^2$]', r'U$\rm_{FK5}$ [mJy/arcsec$^2$]']

    fig, ax = plt.subplots(figsize=(16, 4), ncols=len(keys))

    for ii, d in enumerate(keys):
        plotmap(a.c_radec, d, bins=a.c_radecbins, title=name[ii],
                fig=fig, ax=ax[ii], vmax=np.nanmax(d)*0.2, vmin=np.nanmin(d)*0.2)

    fig.tight_layout()

    fig.savefig(name_format+"_iqu.pdf")
    plt.clf()
    plt.close()


    # In[10]:


    keys = [a.gi*fcf, a.gq*fcf, a.gu*fcf]
    name = [r'I$\rm_{GAL}$ [mJy/arcsec$^2$]', r'Q$\rm_{GAL}$ [mJy/arcsec$^2$]', r'U$\rm_{GAL}$ [mJy/arcsec$^2$]']

    fig, ax = plt.subplots(figsize=(16, 4), ncols=len(keys))

    for ii, d in enumerate(keys):
        plotmap(a.c_glgb, d, bins=a.c_glgbbins, title=name[ii],
                fig=fig, ax=ax[ii], vmax=np.nanmax(d)*0.2, vmin=np.nanmin(d)*0.2)

    fig.tight_layout()

    fig.savefig(name_format+"_galiqu.pdf")
    plt.clf()
    plt.close()


    # In[11]:


    amp = scuba2tool.polamp(a.q, a.u)
    ang = scuba2tool.polang(a.q, a.u)

    keys = [amp*fcf, ang]
    name = [r'PolAmpl$\rm_{FK5}$ [mJy/arcsec$^2$]', r'PolAngle$\rm_{FK5}$ [deg]']
    maxv = [0.8, 180]

    fig, ax = plt.subplots(figsize=(11, 4), ncols=len(keys))

    for d in keys:
        sigma = np.abs(a.i/a.di)
        d[sigma < 2] = 0

    for ii, d in enumerate(keys):
        plotmap(a.c_radec, d, bins=a.c_radecbins, title=name[ii],
                fig=fig, ax=ax[ii], vmax=maxv[ii], vmin=0)

    fig.tight_layout()

    fig.savefig(name_format+"_pol.pdf")
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 4))

    plotmap(a.c_radec, scuba2tool.diqu(a.di, a.dq, a.du)*fcf, bins=a.c_radecbins,
            title=r'Total RMS$\rm_{FK5}$ [mJy/arcsec$^2$]',
            fig=fig, ax=ax, vmin=0, vmax=0.3)

    fig.tight_layout()

    fig.savefig(name_format+"_rms.pdf")
    plt.clf()
    plt.close()

    # In[12]:


    gamp = scuba2tool.polamp(a.gq, a.gu)
    gang = scuba2tool.polang(a.gq, a.gu)

    keys = [gamp*fcf, gang]
    name = [r'PolAmpl$\rm_{GAL}$ [mJy/arcsec$^2$]', r'PolAngle$\rm_{GAL}$ [deg]']
    maxv = [0.8, 180]

    fig, ax = plt.subplots(figsize=(11, 4), ncols=len(keys))

    for d in keys:
        sigma = np.abs(a.i/a.di)
        d[sigma < 2] = 0

    for ii, d in enumerate(keys):
        plotmap(a.c_glgb, d, bins=a.c_glgbbins, title=name[ii],
                fig=fig, ax=ax[ii], vmax=maxv[ii], vmin=0)

    fig.tight_layout()

    fig.savefig(name_format+"_galpol.pdf")
    plt.clf()
    plt.close()


    # In[13]:


    # keys = [a.i*fcf, a.q*fcf, a.u*fcf, amp*fcf, ang]
    # name = [r'I$\rm_{FK5}$ [mJy/arcsec$^2$]', r'Q$\rm_{FK5}$ [mJy/arcsec$^2$]', r'U$\rm_{FK5}$ [mJy/arcsec$^2$]', r'PolAmpl$\rm_{FK5}$ [mJy/arcsec$^2$]', r'PolAngle$\rm_{FK5}$ [deg]']
    # maxv = [0.8, 0.8, 0.8, 0.8, 180]

    # fig, ax = plt.subplots(figsize=(6*len(keys), 4), ncols=len(keys))

    # for d in keys:
    #     sigma = np.abs(a.i/a.di)
    #     d[sigma < 2] = 0

    # for ii, d in enumerate(keys[:3]):
    #     plotmap(a.c_radec, d, bins=a.c_radecbins, title=name[ii],
    #             fig=fig, ax=ax[ii], vmax=np.nanmax(d)*0.2, vmin=np.nanmin(d)*0.2)

    # for ii, d in enumerate(keys[3:]):
    #     plotmap(a.c_radec, d, bins=a.c_radecbins, title=name[ii+3],
    #             fig=fig, ax=ax[ii+3], vmax=maxv[ii+3])

    # for iax in ax:
    #     draw_circle = plt.Circle(radeccent, r_signal,fill=False,color='white')
    #     iax.add_artist(draw_circle)
    #     draw_circle = plt.Circle(radeccent, 0.05,fill=False,color='white')
    #     iax.add_artist(draw_circle)
    #     draw_circle = plt.Circle(radeccent, 0.03,fill=False,color='white')
    #     iax.add_artist(draw_circle)
    #     iax.scatter([cra],[cdec],color='w',marker='x')

    #     draw_circle = plt.Circle(radeccent_l, r_rms, fill=False, color='grey', ls='-', lw=3)
    #     iax.add_artist(draw_circle)
    #     draw_circle = plt.Circle(radeccent_u, r_rms, fill=False, color='grey', ls='-', lw=3)
    #     iax.add_artist(draw_circle)
    #     draw_circle = plt.Circle(radeccent_r, r_rms, fill=False, color='grey', ls='-', lw=3)
    #     iax.add_artist(draw_circle)
    #     draw_circle = plt.Circle(radeccent_b, r_rms, fill=False, color='grey', ls='-', lw=3)
    #     iax.add_artist(draw_circle)

    # fig.tight_layout()

    # fig.savefig(name_format+"_pol_region.pdf")
    # plt.clf()
    # plt.close()


    # # In[14]:


    # keys = [a.i*fcf, a.q*fcf, a.u*fcf, amp*fcf, ang]
    # name = [r'I$\rm_{FK5}$ [mJy/arcsec$^2$]', r'Q$\rm_{FK5}$ [mJy/arcsec$^2$]', r'U$\rm_{FK5}$ [mJy/arcsec$^2$]', r'PolAmpl$\rm_{FK5}$ [mJy/arcsec$^2$]', r'PolAngle$\rm_{FK5}$ [deg]']
    # maxv = [0.8*0.1, 0.8*0.1, 0.8*0.1, 0.8*0.5, 180]

    # fig, ax = plt.subplots(figsize=(6*len(keys), 4), ncols=len(keys))
    # #ax = ax.flatten()

    # for ii, d in enumerate(keys[:3]):
    #     newd = scuba2tool.mask(d,a.astmask)
    #     plotmap(a.c_radec, newd, bins=a.c_radecbins, title=name[ii]+' with astmask',
    #             fig=fig, ax=ax[ii], vmax=np.nanmax(newd)*0.01, vmin=np.nanmin(newd)*0.01)

    # for ii, d in enumerate(keys[3:]):
    #     newd = scuba2tool.mask(d,a.astmask)
    #     plotmap(a.c_radec, newd, bins=a.c_radecbins, title=name[ii+3]+' with astmask',
    #             fig=fig, ax=ax[ii+3], vmax=maxv[ii+3])

    # for iax in ax:
    #     draw_circle = plt.Circle(radeccent, r_signal,fill=False,color='white')
    #     iax.add_artist(draw_circle)
    #     draw_circle = plt.Circle(radeccent, 0.05,fill=False,color='white')
    #     iax.add_artist(draw_circle)
    #     draw_circle = plt.Circle(radeccent, 0.03,fill=False,color='white')
    #     iax.add_artist(draw_circle)
    #     iax.scatter([cra],[cdec],color='w',marker='x')

    #     draw_circle = plt.Circle(radeccent_l, r_rms, fill=False, color='grey', ls='-', lw=3)
    #     iax.add_artist(draw_circle)
    #     draw_circle = plt.Circle(radeccent_u, r_rms, fill=False, color='grey', ls='-', lw=3)
    #     iax.add_artist(draw_circle)
    #     draw_circle = plt.Circle(radeccent_r, r_rms, fill=False, color='grey', ls='-', lw=3)
    #     iax.add_artist(draw_circle)
    #     draw_circle = plt.Circle(radeccent_b, r_rms, fill=False, color='grey', ls='-', lw=3)
    #     iax.add_artist(draw_circle)


    # fig.tight_layout()


    # fig.savefig(name_format+"_pol_region2.pdf")
    # plt.clf()
    # plt.close()


    # # In[15]:

    # ss = ~np.isnan(a.astmask) & ~np.isnan(a.pcamask)
    # def _get(i,j,k):
    #     if not ss[k]:
    #         return np.nan
    #     if i==0 and i==j:
    #         return np.nan
    #     return i-j
    # #d = np.array([int(i)-int(j) if ss[k] else np.nan for k,(i,j) in enumerate(zip(a.astmask,a.pcamask))])
    # d = np.array([_get(i,j,k) for k,(i,j) in enumerate(zip(a.astmask,a.pcamask))])
    # #d[np.isnan(a.astmask) & np.isnan(a.pcamask)]
    # fig,ax = plt.subplots(figsize=(12,4),ncols=2)
    # ax[0].plot(d,'o')
    # ax[0].set_ylabel("astmask - pcamask")
    # ax[0].set_xlabel("number")
    # plotmap(a.c_radec, d, bins=a.c_radecbins, title='astmask - pcamask',fig=fig,ax=ax[1],vmax=1,vmin=-1,count=3)
    # ax[1].set_facecolor('grey')

    # fig.savefig(name_format+"_mask.pdf")
    # plt.clf()
    # plt.close()

    keys = [a.iauto*fcf, a.i*fcf, a.iauto*fcf, a.i*fcf]
    name = [r'(A) I$\rm_{FK5,auto}$ [mJy/arcsec$^2$]', r'(B) I$\rm_{FK5}$ [mJy/arcsec$^2$]', r'(C) I$\rm_{FK5,auto,ASTmasked}$ [mJy/arcsec$^2$]', r'(D) I$\rm_{FK5,ASTmasked}$ [mJy/arcsec$^2$]']

    fig, ax = plt.subplots(figsize=(6*len(keys)+6, 8), ncols=len(keys)+1, nrows=2)
    ax = ax.flatten()

    for ii, d in enumerate(keys):
        if ii>1:
            newd = scuba2tool.mask(d,a.astmask)
            plotmap(a.radec, newd, bins=a.radecbins, title=name[ii],
                    fig=fig, ax=ax[ii], vmax=0.2, vmin=-0.2)
        else:
            newd = d
            plotmap(a.radec, newd, bins=a.radecbins, title=name[ii],
                    fig=fig, ax=ax[ii], vmax=0.2, vmin=-0.2)

    ss = ~np.isnan(a.astmask) & ~np.isnan(a.pcamask)
    def _get(i,j,k):
        if not ss[k]:
            return np.nan
        if i==0 and i==j:
            return np.nan
        return i-j
    d = np.array([_get(i,j,k) for k,(i,j) in enumerate(zip(a.astmask,a.pcamask))])
    plotmap(a.radec, d, bins=a.radecbins, title='(E) astmask - pcamask',fig=fig,ax=ax[4],vmax=1,vmin=-1,count=3)

    name = ["(F) ", "(G) ", "(H) ", "(I) ","(J) "]
    for ii,s in enumerate(asub):
        newd = s.i*fcf
        plotmap(s.c_radec, newd, bins=s.c_radecbins, title=name[ii]+r'I$\rm_{FK5}$ #'+f'{ii}'+r' [mJy/arcsec$^2$]',
                fig=fig, ax=ax[ii+5], vmax=0.2, vmin=-0.2)

    for ii,iax in enumerate(ax):
        if ii == 4:
            continue

        draw_circle = plt.Circle(radeccent, r_signal,fill=False,color='white')
        iax.add_artist(draw_circle)
        draw_circle = plt.Circle(radeccent, 0.05,fill=False,color='white')
        iax.add_artist(draw_circle)

        iax.scatter([cra],[cdec],color='k',marker='x')

        draw_circle = plt.Circle(radeccent_l, r_rms, fill=False, color='grey', ls='-', lw=2)
        iax.add_artist(draw_circle)
        draw_circle = plt.Circle(radeccent_u, r_rms, fill=False, color='grey', ls='-', lw=2)
        iax.add_artist(draw_circle)
        draw_circle = plt.Circle(radeccent_r, r_rms, fill=False, color='grey', ls='-', lw=2)
        iax.add_artist(draw_circle)
        draw_circle = plt.Circle(radeccent_b, r_rms, fill=False, color='grey', ls='-', lw=2)
        iax.add_artist(draw_circle)

    fig.tight_layout()

    fig.savefig(name_format+"_miscI.pdf")
    plt.clf()
    plt.close()

    # In[16]:

    with open(name_format+"_tables.tex", mode='w') as f:
        f.write('\n\n')

    # In[17]:

    # colnames = ['I','Q','U','PolAmpl','PolAngle','PolFrac','N','R']

    # df = pd.DataFrame(index=[],columns=colnames)

    # fig, ax = plt.subplots(ncols=7, figsize=(28, 4))

    # imask = scuba2tool.mask(a.i,a.astmask)
    # qmask = scuba2tool.mask(a.q,a.astmask)
    # umask = scuba2tool.mask(a.u,a.astmask)

    # avg = [np.nan]*8

    # l = scuba2integral.calc_mean(imask,qmask,umask, a.c_radec, radeccent_l, r_rms, fcf=fcf,
    #                              fig=fig,ax=ax,c='b',un="[mJy/arcsec2]", do_print=False,label='L')
    # record = pd.Series(l, index=df.columns, name='L')
    # df = df.append(record)
    # avg = np.nanmean((avg,l),axis=0)

    # l = scuba2integral.calc_mean(imask,qmask,umask, a.c_radec, radeccent_r, r_rms, fcf=fcf,
    #                              fig=fig,ax=ax,c='cyan',un="[mJy/arcsec2]", do_print=False,label='R')
    # record = pd.Series(l, index=df.columns, name='R')
    # df = df.append(record)
    # avg = np.nanmean((avg,l),axis=0)

    # l = scuba2integral.calc_mean(imask,qmask,umask, a.c_radec, radeccent_u, r_rms, fcf=fcf,
    #                              fig=fig,ax=ax,c='r',un="[mJy/arcsec2]", do_print=False,label='U')
    # record = pd.Series(l, index=df.columns, name='U')
    # df = df.append(record)
    # avg = np.nanmean((avg,l),axis=0)

    # l = scuba2integral.calc_mean(imask,qmask,umask, a.c_radec, radeccent_b, r_rms, fcf=fcf,
    #                              fig=fig,ax=ax,c='orange',un="[mJy/arcsec2]", do_print=False,label='B')
    # record = pd.Series(l, index=df.columns, name='B')
    # df = df.append(record)
    # avg = np.nanmean((avg,l),axis=0)

    # #record = pd.Series(avg, index=df.columns, name='ALL')
    # df = df.append(record)

    # for iax in ax:
    #     iax.axvline(r_rms*3600,ls=':',c='k')

    # fig.savefig(name_format+"_integ_mean.pdf")
    # plt.clf()
    # plt.close()

    # for ii,s in enumerate(asub):
    #     fig, ax = plt.subplots(ncols=7, figsize=(28, 4))
    #     avg = [np.nan]*8
    #     imask = scuba2tool.mask(s.i,s.astmask)
    #     qmask = scuba2tool.mask(s.q,s.astmask)
    #     umask = scuba2tool.mask(s.u,s.astmask)

    #     l = scuba2integral.calc_mean(imask,qmask,umask, a.c_radec, radeccent_l, r_rms, fcf=fcf,
    #                                  fig=fig,ax=ax,c='b',un="[mJy/arcsec2]", do_print=False,label='L')
    #     record = pd.Series(l, index=df.columns, name=f'L_{ii}')
    #     df = df.append(record)
    #     avg = np.nanmean((avg,l),axis=0)

    #     l = scuba2integral.calc_mean(imask,qmask,umask, a.c_radec, radeccent_r, r_rms, fcf=fcf,
    #                                  fig=fig,ax=ax,c='cyan',un="[mJy/arcsec2]", do_print=False,label='R')
    #     record = pd.Series(l, index=df.columns, name=f'R_{ii}')
    #     df = df.append(record)
    #     avg = np.nanmean((avg,l),axis=0)

    #     l = scuba2integral.calc_mean(imask,qmask,umask, a.c_radec, radeccent_u, r_rms, fcf=fcf,
    #                                  fig=fig,ax=ax,c='r',un="[mJy/arcsec2]", do_print=False,label='U')
    #     record = pd.Series(l, index=df.columns, name=f'U_{ii}')
    #     df = df.append(record)
    #     avg = np.nanmean((avg,l),axis=0)

    #     l = scuba2integral.calc_mean(imask,qmask,umask, a.c_radec, radeccent_b, r_rms, fcf=fcf,
    #                                  fig=fig,ax=ax,c='orange',un="[mJy/arcsec2]", do_print=False,label='B')
    #     record = pd.Series(l, index=df.columns, name=f'B_{ii}')
    #     df = df.append(record)
    #     avg = np.nanmean((avg,l),axis=0)

    #     #record = pd.Series(avg, index=df.columns, name=f'ALL_{ii}')
    #     df = df.append(record)

    #     pass

    # ltx_str = df.to_latex(float_format="%.4g")

    # print("============ blank region / mean ============")
    # print(df)

    # with open(name_format+"_tables.tex", mode='a') as f:
    #     f.write(ltx_str+'\n\n')


    # In[18]:

    # colnames = ['I','Q','U','PolAmpl','PolAngle','PolFrac','N','R']

    # df = pd.DataFrame(index=[],columns=colnames)

    # fig, ax = plt.subplots(ncols=7, figsize=(28, 4))

    # imask = scuba2tool.mask(a.i,a.astmask)
    # qmask = scuba2tool.mask(a.q,a.astmask)
    # umask = scuba2tool.mask(a.u,a.astmask)

    # l = scuba2integral.calc_rms(imask,qmask,umask, a.c_radec, radeccent_l, r_rms, dist1=r_rms*2, fcf=fcf,
    #                             fig=fig,ax=ax,c='b',un="[mJy/arcsec2]", do_print=False,label='L')
    # record = pd.Series(l, index=df.columns, name='L')
    # df = df.append(record)

    # l = scuba2integral.calc_rms(imask,qmask,umask, a.c_radec, radeccent_r, r_rms, dist1=r_rms*2, fcf=fcf,
    #                              fig=fig,ax=ax,c='cyan',un="[mJy/arcsec2]", do_print=False,label='R')
    # record = pd.Series(l, index=df.columns, name='R')
    # df = df.append(record)

    # l = scuba2integral.calc_rms(imask,qmask,umask, a.c_radec, radeccent_u, r_rms, dist1=r_rms*2, fcf=fcf,
    #                              fig=fig,ax=ax,c='r',un="[mJy/arcsec2]", do_print=False,label='U')
    # record = pd.Series(l, index=df.columns, name='U')
    # df = df.append(record)

    # l = scuba2integral.calc_rms(imask,qmask,umask, a.c_radec, radeccent_b, r_rms, dist1=r_rms*2, fcf=fcf,
    #                              fig=fig,ax=ax,c='orange',un="[mJy/arcsec2]", do_print=False,label='B')
    # record = pd.Series(l, index=df.columns, name='B')
    # df = df.append(record)

    # for iax in ax:
    #     iax.axvline(r_rms*3600,ls=':',c='k')

    # fig.savefig(name_format+"_integ_rms.pdf")
    # plt.clf()
    # plt.close()

    # for ii,s in enumerate(asub):
    #     fig, ax = plt.subplots(ncols=7, figsize=(28, 4))
    #     imask = scuba2tool.mask(s.i,s.astmask)
    #     qmask = scuba2tool.mask(s.q,s.astmask)
    #     umask = scuba2tool.mask(s.u,s.astmask)

    #     l = scuba2integral.calc_rms(imask,qmask,umask, a.c_radec, radeccent_l, r_rms, fcf=fcf,
    #                                  fig=fig,ax=ax,c='b',un="[mJy/arcsec2]", do_print=False,label='L')
    #     record = pd.Series(l, index=df.columns, name=f'L_{ii}')
    #     df = df.append(record)

    #     l = scuba2integral.calc_rms(imask,qmask,umask, a.c_radec, radeccent_r, r_rms, fcf=fcf,
    #                                  fig=fig,ax=ax,c='cyan',un="[mJy/arcsec2]", do_print=False,label='R')
    #     record = pd.Series(l, index=df.columns, name=f'R_{ii}')
    #     df = df.append(record)

    #     l = scuba2integral.calc_rms(imask,qmask,umask, a.c_radec, radeccent_u, r_rms, fcf=fcf,
    #                                  fig=fig,ax=ax,c='r',un="[mJy/arcsec2]", do_print=False,label='U')
    #     record = pd.Series(l, index=df.columns, name=f'U_{ii}')
    #     df = df.append(record)

    #     l = scuba2integral.calc_rms(imask,qmask,umask, a.c_radec, radeccent_b, r_rms, fcf=fcf,
    #                                  fig=fig,ax=ax,c='orange',un="[mJy/arcsec2]", do_print=False,label='B')
    #     record = pd.Series(l, index=df.columns, name=f'B_{ii}')
    #     df = df.append(record)

    #     pass

    # ltx_str = df.to_latex(float_format="%.4g")

    # print("============ blank region / rms  ============")
    # print(df)

    # with open(name_format+"_tables.tex", mode='a') as f:
    #     f.write(ltx_str+'\n\n')


    colnames = ['$I$','$Q$','$U$','$I_p$','$N$','$R$']

    df = pd.DataFrame(index=[],columns=colnames)

    fig, ax = plt.subplots(ncols=5, figsize=(20, 4))

    imask = scuba2tool.mask(a.i,a.astmask)
    qmask = scuba2tool.mask(a.q,a.astmask)
    umask = scuba2tool.mask(a.u,a.astmask)

    l = scuba2integral.calc_noise(imask,qmask,umask, a.c_radec, radeccent_l, r_rms, dist1=r_rms*2, fcf=fcf,
                                  fig=fig,ax=ax,c='b',un="[mJy/arcsec2]", do_print=False,label='L')
    record = pd.Series(l, index=df.columns, name='$L$')
    df = df.append(record)

    l = scuba2integral.calc_noise(imask,qmask,umask, a.c_radec, radeccent_r, r_rms, dist1=r_rms*2, fcf=fcf,
                                 fig=fig,ax=ax,c='cyan',un="[mJy/arcsec2]", do_print=False,label='R')
    record = pd.Series(l, index=df.columns, name='$R$')
    df = df.append(record)

    l = scuba2integral.calc_noise(imask,qmask,umask, a.c_radec, radeccent_u, r_rms, dist1=r_rms*2, fcf=fcf,
                                 fig=fig,ax=ax,c='r',un="[mJy/arcsec2]", do_print=False,label='U')
    record = pd.Series(l, index=df.columns, name='$U$')
    df = df.append(record)

    l = scuba2integral.calc_noise(imask,qmask,umask, a.c_radec, radeccent_b, r_rms, dist1=r_rms*2, fcf=fcf,
                                 fig=fig,ax=ax,c='orange',un="[mJy/arcsec2]", do_print=False,label='B')
    record = pd.Series(l, index=df.columns, name='$B$')
    df = df.append(record)

    for iax in ax:
        iax.axvline(r_rms*3600,ls=':',c='k')

    fig.savefig(name_format+"_integ_rms.pdf")
    plt.clf()
    plt.close()

    for ii,s in enumerate(asub):
        fig, ax = plt.subplots(ncols=5, figsize=(20, 4))
        imask = scuba2tool.mask(s.i,s.astmask)
        qmask = scuba2tool.mask(s.q,s.astmask)
        umask = scuba2tool.mask(s.u,s.astmask)

        l = scuba2integral.calc_noise(imask,qmask,umask, s.c_radec, radeccent_l, r_rms, fcf=fcf,
                                      fig=fig,ax=ax,c='b',un="[mJy/arcsec2]", do_print=False,label='L')
        record = pd.Series(l, index=df.columns, name=f'$L_{ii}$')
        df = df.append(record)

        l = scuba2integral.calc_noise(imask,qmask,umask, s.c_radec, radeccent_r, r_rms, fcf=fcf,
                                      fig=fig,ax=ax,c='cyan',un="[mJy/arcsec2]", do_print=False,label='R')
        record = pd.Series(l, index=df.columns, name=f'$R_{ii}$')
        df = df.append(record)

        l = scuba2integral.calc_noise(imask,qmask,umask, s.c_radec, radeccent_u, r_rms, fcf=fcf,
                                      fig=fig,ax=ax,c='r',un="[mJy/arcsec2]", do_print=False,label='U')
        record = pd.Series(l, index=df.columns, name=f'$U_{ii}$')
        df = df.append(record)

        l = scuba2integral.calc_noise(imask,qmask,umask, s.c_radec, radeccent_b, r_rms, fcf=fcf,
                                      fig=fig,ax=ax,c='orange',un="[mJy/arcsec2]", do_print=False,label='B')
        record = pd.Series(l, index=df.columns, name=f'$B_{ii}$')
        df = df.append(record)

        pass

    ltx_str = df.to_latex(float_format="%.4g")

    print("============ blank region / rms  ============")
    print(df)

    with open(name_format+"_tables.tex", mode='a') as f:
        f.write(ltx_str+'\n\n')

    # In[19]:


    colnames = ['$I$','$Q$','$U$','$I_p$','$\psi_p$', '$p$', '$N$', '$R$']
    df = pd.DataFrame(index=[],columns=colnames)

    fig, ax = plt.subplots(ncols=7, figsize=(28, 4))

    l = scuba2integral.calc_integration(*a.iqu, *a.diqu,
                                        a.c_radec, radeccent, r_signal, r_signal*1.5,fcf=fcf,
                                        fig=fig,ax=ax,c='b',un="[mJy/arcsec2]",do_print=False)
    record = pd.Series(l, index=df.columns, name='FK5')
    df = df.append(record)

    for iax in ax:
        iax.axvline(r_signal*3600,ls=':',c='k')

    fig.savefig(name_format+"_integ_radec.pdf")
    plt.clf()
    plt.close()

    for ii,s in enumerate(asub):
        fig, ax = plt.subplots(ncols=7, figsize=(28, 4))
        l = scuba2integral.calc_integration(*s.iqu, *s.diqu,
                                            s.c_radec, radeccent, r_signal, r_signal*1.5,fcf=fcf,
                                            fig=fig,ax=ax,c='b',un="[mJy/arcsec2]",do_print=False)
        record = pd.Series(l, index=df.columns, name=f'FK5$_{ii}$')
        df = df.append(record)
        pass

    # In[20]:

    fig, ax = plt.subplots(ncols=7, figsize=(28, 4))

    l = scuba2integral.calc_integration(a.gi,a.gq,a.gu, a.gdi,a.gdq,a.gdu, 
                                        a.c_glgb, glgbcent, r_signal, r_signal*1.5,fcf=fcf,
                                        fig=fig,ax=ax,c='b',un="[mJy/arcsec2]",do_print=False)
    record = pd.Series(l, index=df.columns, name='GAL')
    df = df.append(record)

    for iax in ax:
        iax.axvline(r_signal*3600,ls=':',c='k')

    fig.savefig(name_format+"_integ_gal.pdf")
    plt.clf()
    plt.close()

    ltx_str = df.to_latex(float_format="%.6g")

    print("============ signal region / integration  ============")
    print(df)
    print(f"INTEGRATED NPIX = {l[-2]}")

    with open(name_format+"_tables.tex", mode='a') as f:
        f.write(ltx_str+'\n\n')


    # In[ ]:

if __name__ == '__main__':

    args = sys.argv
    if len(args)!=3:
        print("ERROR:: run.py [DIRECTORY] [NAME_FORMAT]")
        pass

    fdir = args[1] #"star21_850um_customPca/pca50/"
    name_format = args[2] #"pca50_850um"
    os.makedirs(name_format,exist_ok=True)
    name_format = name_format + "/" + name_format

    main(fdir,name_format)

