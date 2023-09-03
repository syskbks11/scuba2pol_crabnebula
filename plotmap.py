
import numpy as np
import copy

from astropy import units

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams.update({'font.size': 14})
mpl.rcParams.update({'axes.facecolor': 'w'})
mpl.rcParams.update({'axes.edgecolor': 'k'})
mpl.rcParams.update({'figure.facecolor': 'w'})
mpl.rcParams.update({'figure.edgecolor': 'w'})
mpl.rcParams.update({'axes.grid': True})
mpl.rcParams.update({'grid.linestyle': ':'})
mpl.rcParams.update({'figure.figsize': [12, 9]})

def wcs_subplots(wcs=None, nrows=1, ncols=1, **fig_kw):
    '''
    make figure/axis for mapplot with wcs
    '''
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
    '''
    draw circle on wcsaxes plot
    '''
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
    '''
    draw point on wcsaxes plot
    '''
    if not isinstance(xy,units.quantity.Quantity):
        xy = xy*units.degree
    if isinstance(transform,str):
        transform = ax.get_transform(transform)

    ax.scatter(*xy, transform=transform,  **kwargs)
    return ax

def wcsaxes_lim(xlim,ylim,ax,wcs):
    '''
    set axis limit on wcsaxes plot
    '''
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

def plotmap(x, wcs, title='', fig=None, ax=None,
            ax_decimal=False, ax_galactic=False, ax_fk5=False,count=None, plot_contour=False,
            xlim=(None, None), ylim=(None, None),
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

    # if nikalim:
    #     if wcs.axis_type_names[0] == 'RA':
    #         xlim,ylim = get_axeslim_nika('fk5')
    #         wcsaxes_lim(xlim,ylim,ax,wcs)
    #     elif wcs.axis_type_names[0] == 'GLON':
    #         xlim,ylim = get_axeslim_nika('gal')
    #         wcsaxes_lim(xlim,ylim,ax,wcs)
    #     else:
    #         print(f"ERROR:: Unknown axis type name in this wcs : {wcs.axis_type_names}")
    #         pass

    # if scuba2lim:
    #     if wcs.axis_type_names[0] == 'RA':
    #         xlim,ylim = get_axeslim_scuba2('fk5')
    #         wcsaxes_lim(xlim,ylim,ax,wcs)
    #     elif wcs.axis_type_names[0] == 'GLON':
    #         xlim,ylim = get_axeslim_scuba2('gal')
    #         wcsaxes_lim(xlim,ylim,ax,wcs)
    #     else:
    #         print(f"ERROR:: Unknown axis type name in this wcs : {wcs.axis_type_names}")
    #         pass

    if not (xlim[0] is None and xlim[1] is None and ylim[0] is None and ylim[1] is None):
        xlim = list(xlim)
        ylim = list(ylim)
        wcsaxes_lim(xlim,ylim,ax,wcs)
        pass

    return fig, ax

