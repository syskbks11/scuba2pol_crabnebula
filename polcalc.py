import numpy as np
import healpy as hp

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

