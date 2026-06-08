#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Useful functions.
"""
import numpy as np
import pandas as pd
import datetime
from astroquery.jplhorizons import Horizons
from decimal import Decimal, ROUND_HALF_UP
import sep

# elevation in km
loc_Pirka = "Q33"
loc_Subaru = "T09"
loc_NOT = "Z23"
# NHAO: http://www.nhao.jp/research/nayuta_telescope.html
loc_Nayuta = {
    "lon": 134.3356,
    "lat": 35.0253,
    "elevation": 0.449
    }
# Higashi-Hiroshima: https://www.hiroshima-u.ac.jp/hasc/hho_kanata
loc_Kanata = {
    "lon": 132.7767,
    "lat": 34.3775,
    "elevation": 0.5112
    }


def utc2alphaphi(obj, ut, loc):
    """
    Return phase angle and position angle of scattering plane 
    with JPL/Horizons.

    Parameters
    ----------
    obj : str
        object name
    ut : str
        utc like "2022-12-21T15:53:07.3"
    loc : str
        location of the observatory, MPC code or dictinary (lon, lat, elevation) 
    """
    try:
        t0_dt = datetime.datetime.strptime(ut, "%Y-%m-%dT%H:%M:%S.%f")
    except:
        t0_dt = datetime.datetime.strptime(ut, "%Y-%m-%dT%H:%M:%S")
    t1_dt = t0_dt + datetime.timedelta(minutes=5)
    t0 = datetime.datetime.strftime(t0_dt, "%Y-%m-%dT%H:%M:%S.%f")
    t1 = datetime.datetime.strftime(t1_dt, "%Y-%m-%dT%H:%M:%S.%f")
    jpl = Horizons(id=obj, location=loc,
        epochs={'start':t0, 'stop':t1, 'step':"1m"})
    eph = jpl.ephemerides()
    alpha  = eph[0]["alpha"]
    phi    = eph[0]["sunTargetPA"]

    return alpha, phi


def adderr(*args):
    """Calculate additional error.

    Parameters
    ----------
    args : array-like
        list of values

    Return
    ------
    err : float
        calculated error
    """
    err = np.sqrt(np.sum(np.square(args)))
    return err


def diverr(val1, err1, val2, err2):
    """Calculate error for division.
    
    Parameters
    ----------
    val1 : float or pandas.Series 
      value 1
    err1 : float or pandas.Series 
      error 1
    val2 : float or pandas.Series 
      value 2
    err2 : float or pandas.Series 
      error 2
    """
    return np.sqrt((err1/val2)**2 + (err2*val1/val2**2)**2)


def count_length(value):
    """Count float part length of digit.
    Example:
      12    -> 2
      9     -> 1
      0.1   -> -1
      0.023 -> -2

    Parameter 
    ---------
    Value : float
        input value
    
    Return 
    ------
    l : int
        length of float part
    """
    n = 0
    if value < 1.:
        while value < 1.:
            value = value*10.
            n = n-1

    elif value >= 1.:
        while value >= 1.:
            value = value/10.
            n = n+1
    return n


def round_error(value, err):
    """
    Round a value using its error.
    0 is added at the end, for example,  when (value, error) = (120, 0.1).
    Then returned value and error are 120.0, 0.1.

    Parameters
    ----------
    value : float
        value
    err : float
        error of the value

    Returns
    -------
    value_round : str
        rounded value
    err_round : str
        rounded error
    """
    if err >= 1.:
        # Ex: value = 212
        #       err = 51
        #  a = 2
        a = count_length(err)
        # Convert error to float temporally
        # err = 51/10 = 5.1
        err = err/10**(a-1)
        # err_round = 5
        err_round = Decimal(str(err)).quantize(Decimal("1"), ROUND_HALF_UP)
        # Re-conver
        # err_round = 50
        err_round = err_round*10**(a-1)
        
        # value = 21.2
        value = value/10**(a-1)
        # value_round = 21
        value_round = Decimal(str(value)).quantize(Decimal("1"), ROUND_HALF_UP)
        # value_round = 210
        value_round = value_round*10**(a-1)

    elif 0 < err < 1.:
        # Ex: value = 21.2
        #       err = 0.032
        #  a = -2
        a = count_length(err)
        # if a=2, unit=0.01 and value is rounded to 0.0X using 0.00X value
        # ex) value=0.042, a=2, unit=0.01, and output is 0.04
        # unit = 0.01
        unit = 10**(a)
        # err_round = 0.03
        err_round = Decimal(
            str(err)).quantize(Decimal(str(unit)), ROUND_HALF_UP)
        
        # Useless ? ===========================================================
        # Remove 0 at the end
        if str(err_round)[-1]=="0":
            # ex) 0.010 to 0.01
            err_round = str(err_round)[:-1]
            # Remove a 0 
            unit = 10**(a+1)
        # =====================================================================


      #  # Useless ? for (val, err) = (15.2881, 0.0099) =======================
      #  # Remove 0 at the end
      #  if str(err_round)[-1]=="0":
      #    # ex) 0.010 to 0.01
      #    err_round = str(err_round)[:-1]
      #    # ex) 0.001 to 0.01
      #    unit = unit[:-2] + "1"
      #    assert False, unit
      #  # ====================================================================

        value_round = Decimal(
          str(value)).quantize(Decimal(str(unit)), ROUND_HALF_UP)
    
    elif err==0:
        value_round = value
        err_round = err
    else:
        print(err)
        assert False, f"Negative or nan in erorr!"

    value_round = str(value_round)
    err_round   = str(err_round)
    return value_round, err_round


# Photometry ==================================================================
def remove_bg_2d(image, mask=None, bw=64, fw=3):
    """ Remove background from 2D FITS

    Parameters
    ----------
    image : array-like
        input image to be background subtracted
    mask : array-like
        mask array 
    bw : int
        box width 
    fw : int
        filter width 

    Returns
    -------
    image : array like
        background subtracted image
    bg_info : dict
        background info.

    """
    bg_engine = sep.Background(
        image, mask=mask, bw=bw, bh=bw, fw=fw, fh=fw)
    bg_engine.subfrom(image)
    bg_global = bg_engine.globalback
    bg_rms = bg_engine.globalrms
    bg_info = {'level': bg_global, 'rms': bg_rms}
    bg = bg_engine.back()
    return image, bg_info


def obtain_winpos(data, x, y, radius, nx, ny):
    """ 
    Obtain windowed centroid xwin and ywin.
    Note: x and y should be numpy.ndarray

    Parameters
    ----------
    data : numpy.ndarray
      2-d image data
    x, y : numpy.ndarray
      location(s) of object(s)
    radius : float
      scale related to the area where searched the centroid
    """

    # Radius should be large enough to obtain total flux
    # wpos_param: constant to convert `0.5*FWHM` to `sigma` (FWHM = 2.35*sigma) 
    # 0.5*FWHM*wpos_param = 0.5*FWHM*2/2.35 = sigma
    wpos_param  = 2.0/2.35
    # Half flux radius
    frad_frac   = 0.5
    frad_subpix = 5
    frad_ratio  = 5.0

    # Do photometry to obtain all flux
    # Note1: err and gain are needless for flux estimation
    # Note2: Sky background should be subtracted (?) 

    # Must need
    flux,fluxerr,eflag = sep.sum_circle(data, x, y, r=radius)

    # Use only objects with nonzero eflag (?)
    # Create array like [radius, radius, ... , radius]
    # Note: radius is not used in flux_radius when normflux is used (?)
    # r : flux radius, i.e., `0.5*fwhm!`
    radius = np.full_like(x, radius)
    r, flag = sep.flux_radius(
        data, x, y, radius, frad_frac, normflux=flux, subpix=frad_subpix)
    # r (0.5*FWHM) to sigma (see above)
    sigma      = wpos_param*r
    sigma_mean = np.mean(sigma)
    sigma_std  = np.std(sigma)
    #print(f"  N={len(sigma)}, calculated in obtain_winpos")
    #print(f"  sigma: {sigma_mean:.1f}+-{sigma_std:.1f}")
    #print(f"  FWHM : {2.35*sigma_mean:.1f} (sigma times 2.35)")

    # wflag is always 0 if mask=None
    # Search winpos with estimated sigma

    # Search narrow region
    # sigma = 0.5*sigma
    # Dramatically works bad for faint objects
    xwin, ywin, wflag = sep.winpos(data, x, y, sigma)

    
    # If the differences of coordinates are larger than ratio_diff*radius,
    # for objects more than ratio_obj*N_obj,
    # print a warning message.
    # original values as xwin and ywin with eflag_win = 1
    ratio_diff = 0.3
    ratio_obj  = 0.5
    diff = np.sqrt((xwin-x)**2 + (ywin-y)**2)
    # 1 for large diff, 0 for small diff
    flag = np.where(diff > ratio_diff*radius, 1, 0)
    ratio_large_diff= np.sum(flag)/flag.size 
    if ratio_large_diff > ratio_obj:
        print(f"      Large winpos correct ratio detected :{ratio_large_diff:.1f}")
        print(f"      This is just a caution. Please check wcs information etc.")

    # Insert original value when xwin and ywin are outside of FoV
    xwin = [x if (x < nx) and (x > 0) and (y < ny) and (y > 0) else x0 for x,y,x0 in zip(xwin,ywin,x)]
    ywin = [y if (x < nx) and (x > 0) and (y < ny) and (y > 0) else y0 for x,y,y0 in zip(xwin,ywin,y)]

    return xwin, ywin, flag
# Photometry ==================================================================

