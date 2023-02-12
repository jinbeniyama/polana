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
from scipy.stats import sigmaclip
import sep


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
        location of the observatory (MPC code)
    """
    t0_dt = datetime.datetime.strptime(ut, "%Y-%m-%dT%H:%M:%S.%f")
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
def remove_background2d_pol(image, mask=None):
    """ Remove background from 2D FITS
    """
    bg_engine = sep.Background(image, mask=mask)
    bg_engine.subfrom(image)
    bg_global = bg_engine.globalback
    bg_rms = bg_engine.globalrms
    bg_info = {'level': bg_global, 'rms': bg_rms}
    bg = bg_engine.back()
    return image, bg_info


def extract(image, sigma, minarea, err, mask, swapped=True, logger=None):
    """ extract objects from the image
    """
    if mask is None: mask = np.zeros_like(image)
    if not swapped: image = image.byteswap().newbyteorder()

    objects = sep.extract(
        image, sigma, minarea=minarea, err=err, mask=mask)
    n_obj = len(objects)
    return objects
# Photometry ==================================================================
