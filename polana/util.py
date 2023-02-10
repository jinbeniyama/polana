#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import os 
import subprocess
from argparse import ArgumentParser as ap
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import datetime
from astroquery.jplhorizons import Horizons
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy
from scipy.stats import sigmaclip
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import sep
from decimal import Decimal, ROUND_HALF_UP

# Angle
ang = np.arange(0, 360, 22.5)
f_o_Dipol2 = [f"flux_o_{int(x*10):04d}" for x in ang]
ferr_o_Dipol2 = [f"fluxerr_o_{int(x*10):04d}" for x in ang]
f_e_Dipol2 = [f"flux_e_{int(x*10):04d}" for x in ang]
ferr_e_Dipol2 = [f"fluxerr_e_{int(x*10):04d}" for x in ang]
col_Dipol2 = f_o_Dipol2 + ferr_o_Dipol2 + f_e_Dipol2 + ferr_e_Dipol2


def utc2alpha(obj, ut, loc):
    """
    Return phase angle with JPL/Horizons.

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

    return alpha


def polana_4angle(df):
    """
    Calculate linear polarization degree and position angle 
    with data in four angles of Half-wave plate (0000, 0225, 0450, 0675).

    Parameter
    ---------
    df : pandas.DataFrame
        input dataframe

    Return
    ------
    P : float
        liner polarization degree
    Perr : float
        1-sigma uncertainty of polarization degree
    theta : float
        position angle of polarization
    thetaerr : float
        1-sigma uncertainty of the position angle 
    """
    df_0000 = df[df["angle"]=="0000"].reset_index()
    df_0225 = df[df["angle"]=="0225"].reset_index()
    df_0450 = df[df["angle"]=="0450"].reset_index()
    df_0675 = df[df["angle"]=="0675"].reset_index()

    f_o_0000    = df_0000.at[0, "flux_o"]
    ferr_o_0000 = df_0000.at[0, "fluxerr_o"]
    f_e_0000    = df_0000.at[0, "flux_e"]
    ferr_e_0000 = df_0000.at[0, "fluxerr_e"]

    f_o_0225    = df_0225.at[0, "flux_o"]
    ferr_o_0225 = df_0225.at[0, "fluxerr_o"]
    f_e_0225    = df_0225.at[0, "flux_e"]
    ferr_e_0225 = df_0225.at[0, "fluxerr_e"]

    f_o_0450    = df_0450.at[0, "flux_o"]
    ferr_o_0450 = df_0450.at[0, "fluxerr_o"]
    f_e_0450    = df_0450.at[0, "flux_e"]
    ferr_e_0450 = df_0450.at[0, "fluxerr_e"]

    f_o_0675    = df_0675.at[0, "flux_o"]
    ferr_o_0675 = df_0675.at[0, "fluxerr_o"]
    f_e_0675    = df_0675.at[0, "flux_e"]
    ferr_e_0675 = df_0675.at[0, "fluxerr_e"]
    
    # See Kawakami+2021, SAG
    # Calculate Rq and Ru
    Rq = np.sqrt(
        (f_e_0000/f_o_0000)/(f_e_0450/f_o_0450)
        )
    Rqerr = np.sqrt(
        1./4.*(
            (f_o_0450/(f_e_0000*f_o_0000*f_e_0450))*ferr_e_0000**2
            + ((f_e_0000*f_o_0450)/(f_e_0450*f_o_0000**3))*ferr_o_0000**2
            + (f_o_0000/(f_e_0000*f_o_0450*f_e_0450))*ferr_o_0450**2
            + ((f_e_0000*f_o_0450)/(f_e_0450**3*f_o_0000))*ferr_e_0450**2
            )
        )
    Ru = np.sqrt(
        (f_e_0225/f_o_0225)/(f_e_0675/f_o_0675)
        )
    Ruerr = np.sqrt(
        1./4.*(
            (f_o_0675/(f_e_0225*f_o_0225*f_e_0675))*ferr_e_0225**2
            + ((f_e_0225*f_o_0675)/(f_e_0675*f_o_0225**3))*ferr_o_0225**2
            + (f_o_0225/(f_e_0225*f_o_0675*f_e_0675))*ferr_o_0675**2
            + ((f_e_0225*f_o_0675)/(f_e_0675**3*f_o_0225))*ferr_e_0675**2
            )
        )

    # Calculate q and u, (q = Q/I and u = U/I)
    # Kawakami+2021, Geem+2022, (not consistent with Ishiguro+2017, Kuroda+2018)
    # But it is ok since we use q**2 and u**2
    q = (1-Rq)/(1+Rq)
    qerr = np.sqrt(
        4/(1+Rq**4)*Rqerr**2
        )
    u = (1-Ru)/(1+Ru)
    uerr = np.sqrt(
        4/(1+Ru**4)*Ruerr**2
        )

    # Calculate P and Perr
    P = np.sqrt(
        q**2 + u**2
        )
    Perr = np.sqrt(
        q**2*qerr**2 + u**2*uerr**2
        )/P

    # Calculate theta (osition angle relative to celestial North pole) 
    # and thetaerr in radian
    # arctan2(u, q) returns arctan(u/q) considering from Q1 to Q4
    theta = 0.5*np.arctan2(u, q)
    thetaerr = 0.5*Perr/P
    # Why slightly different
    # thetaerr = np.sqrt(
    #    0.25/(1+(u/q)**2)**2*((uerr/q)**2 + (u*qerr/q**2)**2)    
    #    )
    return u, uerr, q, qerr, P, Perr, theta, thetaerr



def cor_poleff(
    df, inst, band, key_q="q", key_u="u", key_qerr="qerr", key_uerr="uerr",
    key_q_cor="q_cor", key_u_cor="u_cor", key_qerr_cor="qerr_cor", 
    key_uerr_cor="u_corerr"):
    """
    Do correction about polarization efficiency.

    Parameter
    ---------
    df : pandas.DataFrame
        input dataframe with u, q, etc.
    inst : str
        instrument
    band : str
         filter
    key_q, key_u, key_qerr, key_uerr : str
        keywords for original q, u, and their errors
    key_q_cor, key_u_cor, key_qerr_cor, key_uerr_cor : str
        keywords for corrected q, u, and their errors

    Return
    ------
    df : pandas.DataFrame
        output dataframe with u, q, etc.
    """
    if inst == "MSI":
        # From Ishiguro+2017 (, Geem+2022a) 
        if band == "Rc" or "R":
            # 2015 Values !!
            peff    = 0.9971
            pefferr = 0.0001
            # after 2022-03
            peff    = 0.9955
            pefferr = 0.0001
        if band == "V":
            # 2015 Values !!
            peff    = 0.9967
            pefferr = 0.0003
            # after 2022-03
            peff    = 0.9959
            pefferr = 0.0002

    if inst == "WFGS2":
        # TODO:check
        # From Kawakami+2021
        # In Geem+2022b peff=1 (assumption)
        peff    = 0.982
        pefferr = 0.0

    if inst == "HONIR":
        # TODO:check
        # From Geem+2022b
        peff    = 0.9858
        pefferr = 0.0008

    df[key_q_cor] = df[key_q]/peff
    df[key_u_cor] = df[key_u]/peff 
    df[key_qerr_cor] = df[key_qerr]/peff 
    df[key_uerr_cor] = df[key_uerr]/peff 

    return df


def cor_instpol(
    df, inst, band, key_q="q", key_u="u", key_qerr="qerr", key_uerr="uerr",
    key_q_cor="q_cor", key_u_cor="u_cor", key_qerr_cor="qerr_cor", 
    key_uerr_cor="uerr_cor", key_insrot="insrot"):
    """
    Do correction about instrument polarization.

    Parameter
    ---------
    df : pandas.DataFrame
        input dataframe with u, q, etc.
    inst : str
        instrument
    band : str
         filter
    key_q, key_u, key_qerr, key_uerr : str
        keywords for original q, u, and their errors
    key_q_cor, key_u_cor, key_qerr_cor, key_uerr_cor : str
        keywords for corrected q, u, and their errors
    key_insrot : str
        keyword for angle of instrument rotator

    Return
    ------
    df : pandas.DataFrame
        output dataframe with u, q, etc.
    """


    if inst == "MSI":
        # From Ishiguro+2017 (, Geem+2022a)
        if band == "Rc" or "R":
            # 2015 Values !!
            qinst    = 0.00703
            qinsterr = 0.00033
            uinst    = 0.00337
            uinsterr = 0.00020
            # after 2022-03
            qinst    = 0.00862
            qinsterr = 0.00013
            uinst    = 0.00379
            uinsterr = 0.00013

        elif band == "V":
            # 2015 Values !!
            qinst    = 0.00963
            qinsterr = 0.00029
            uinst    = 0.00453
            uinsterr = 0.00043
            # after 2022-03
            qinst    = 0.01202
            qinsterr = 0.00013
            uinst    = 0.00530
            uinsterr = 0.00013

    if inst == "WFGS2":
        # From code in Geem+2022b
        if band == "Rc" or "R":
            qinst    = -0.00043
            qinsterr =  0.00012
            uinst    = -0.00178
            uinsterr =  0.00011
        if band == "V":
            assert False, "No data."

    if inst == "HONIR":
        # From Geem+2022b
        if band == "Rc" or "R":
            qinst    = -0.000097
            qinsterr =  0.000498
            uinst    = -0.000077
            uinsterr =  0.000371
        if band == "V":
            assert False, "No data."
 
    # In radian
    insrot    = np.deg2rad(df[key_insrot])
    insroterr = 0
    
    df[key_q_cor] =  (
        df[key_q] - np.cos(2*insrot)*qinst + np.sin(2*insrot)*uinst
        )
    df[key_u_cor] =  (
        df[key_u] - np.sin(2*insrot)*qinst - np.cos(2*insrot)*uinst
        )
    df[key_qerr_cor] = np.sqrt(
        df[key_qerr]**2 
        + (np.cos(2*insrot)*qinsterr)**2 + (np.sin(2*insrot)*uinsterr)**2 
        )
    df[key_uerr_cor] = np.sqrt(
        df[key_uerr]**2 
        + (np.sin(2*insrot)*qinsterr)**2 + (np.cos(2*insrot)*uinsterr)**2 
        )

    return df


def cor_paoffset(
    df, inst, band, key_q="q", key_u="u", key_qerr="qerr", key_uerr="uerr",
    key_q_cor="q_cor", key_u_cor="u_cor", key_qerr_cor="qerr_cor", 
    key_uerr_cor="uerr_cor", key_instpa="instpa"):
    """
    Do correction about position angle offset.

    Parameter
    ---------
    df : pandas.DataFrame
        input dataframe with u, q, etc.
    inst : str
        instrument
    band : str
         filter
    key_q, key_u, key_qerr, key_uerr : str
        keywords for original q, u, and their errors
    key_q_cor, key_u_cor, key_qerr_cor, key_uerr_cor : str
        keywords for corrected q, u, and their errors
    key_instpa : str
        keyword for position angle of instrument

    Return
    ------
    df : pandas.DataFrame
        output dataframe with u, q, etc.
    """

    # In degree
    
    # Check the sign carefully !!!
    if inst == "MSI":
        # From Ishiguro+2017 (, Geem+2022a) 3.38
        # But definition of theta_off is different!
        # Here we use theta_off = theta_lt - theta_obs.
        # Thus, -3.38
        if band == "Rc" or "R":
            # 2015 Values !!
            paoffset    = -3.38
            paoffseterr = 0.37
            # after 2022-03
            paoffset    = -3.54
            paoffseterr = 0.11
        if band == "V":
            # 2015 Values !!
            paoffset    = -3.82
            paoffseterr = 0.38
            # after 2022-03
            paoffset    = -3.84
            paoffseterr = 0.14

    if inst == "WFGS2":
        # From Geem+2022b. The value and calculation is ok for this script.
        # But in Geem+2022b, the calculation is inconsistent with offset...?
        paoffset    = -5.19

        # This is ok. Result is similar to that derived using coefficients in
        # Kawakami+2021. (beta = -5.68 deg in Rc)
        #paoffset    = -5.68
        paoffseterr =  0.00

    if inst == "HONIR":
        # From Geem+2022b
        # -36.8 ??? 
        paoffset    = 36.8
        paoffseterr = 0.13
    
    # In degree
    # Here, paoffset = theta_lt - theta_obs,
    # which is different from those defined in Ishigur+2017, Okazaki+2021.
    thetarot    = paoffset - df[key_instpa]
    instpaerr = 0
    thetaroterr = np.sqrt(
        paoffseterr**2 + instpaerr**2
        )

    # In radian
    thetarot    = np.deg2rad(thetarot)
    thetaroterr = np.deg2rad(thetaroterr)
    
    df[key_q_cor] =  (
        np.cos(2*thetarot)*df[key_q] - np.sin(2*thetarot)*df[key_u]
        )
    df[key_u_cor] =  (
        np.sin(2*thetarot)*df[key_q] + np.cos(2*thetarot)*df[key_u]
        )

    df[key_qerr_cor] = np.sqrt(
        (-2*np.sin(2*thetarot)*df[key_q] + 2*np.cos(2*thetarot)*df[key_u])**2*thetaroterr**2
        + (np.cos(2*thetarot)*df[key_qerr])**2 + (np.sin(2*thetarot)*df[key_uerr])**2 
        )
    df[key_uerr_cor] = np.sqrt(
        (-2*np.cos(2*thetarot)*df[key_q] - 2*np.sin(2*thetarot)*df[key_u])**2*thetaroterr**2
        + (-np.sin(2*thetarot)*df[key_qerr])**2 + (np.cos(2*thetarot)*df[key_uerr])**2 
        )
    return df


# def cor_paoffset2(
#     df, inst, key_q="q", key_u="u", key_qerr="qerr", key_uerr="uerr",
#     key_q_cor="q_cor", key_u_cor="u_cor", key_qerr_cor="qerr_cor", 
#     key_uerr_cor="uerr_cor", key_instpa="instpa"):
#     """
#     Do correction about position angle offset.
# 
#     Parameter
#     ---------
#     df : pandas.DataFrame
#         input dataframe with u, q, etc.
#     inst : str
#         instrument
#     key_q, key_u, key_qerr, key_uerr : str
#         keywords for original q, u, and their errors
#     key_q_cor, key_u_cor, key_qerr_cor, key_uerr_cor : str
#         keywords for corrected q, u, and their errors
#     key_instpa : str
#         keyword for position angle of instrument
# 
#     Return
#     ------
#     df : pandas.DataFrame
#         output dataframe with u, q, etc.
#     """
# 
#     if inst == "MSI":
#         # From Ishiguro+2017 (, Geem+2022a) 3.38
#         # But definition of theta_off is different!
#         # Here we use theta_off = theta_lt - theta_obs.
#         # Thus, -3.38
#         paoffset    = -3.38
#         paoffseterr = 0.37
#     if inst == "WFGS2":
#         # From Geem+2022b
#         # in degree
#         paoffset    = -5.19
#         paoffseterr =  0.15
# 
#         # See Kawakami+2021, sag
#         df[key_q_cor] =  0.980*df[key_q] + 0.197*df[key_u]
#         df[key_u_cor] = -0.198*df[key_q] + 0.980*df[key_u]
#         df[key_qerr_cor] = np.sqrt(0.980*df[key_qerr]**2 + 0.197*df[key_uerr]**2)
#         df[key_uerr_cor] = np.sqrt(0.198*df[key_qerr]**2 + 0.980*df[key_uerr]**2)
# 
#     if inst == "HONIR":
#         # From Geem+2022b
#         paoffset    = 36.8
#         paoffseterr = 0.13
#     
#     return df


def calc_Ptheta(
    df, 
    key_P="P", key_theta="theta", key_Perr="Perr", key_thetaerr="thetaerr",
    key_q="q", key_u="u", key_qerr="qerr", key_uerr="uerr"):

    # Calculate P_cor and theta_cor
    df[key_P] = np.sqrt(
        df[key_q]**2 + df[key_u]**2
        )
    df[key_Perr] = np.sqrt(
        df[key_q]**2*df[key_qerr]**2 + df[key_u]**2*df[key_uerr]**2
        )/df[key_P]

    df[key_theta] = 0.5*np.arctan2(df[key_u], df[key_q])

    # 1. standard calculation
    df[key_thetaerr] = np.sqrt(
        0.25/(1+(df[key_u]/df[key_q])**2)**2*((df[key_uerr]/df[key_q])**2 
        + (df[key_u]*df[key_qerr]/df[key_q]**2)**2)    
        )
    # 2. useful result
    df[key_thetaerr] = 0.5*df[key_Perr]/df[key_P]
    # TODO: Why 1. and 2. slightly different ?? (at least MSI data obtained in 2022-12-21)

    return df


def _fit(alpha, H, G):
    """
    Phase curve fitting as
        f(a) = H -2.5lon10((1-G)Phi1 + GPhi2),
    where
        Phi1 = w*(1-C1*sin(a)/(0.119+1.341sin(a) + sin^2(a))) + (1-w)*(-A1tan^B1(a/2)),
        term2 = w*(1-C2*sin(a)/(0.119+1.341sin(a) + sin^2(a))) + (1-w)*(-A2tan^B2(a/2)),
        w = exp(-90.56tan^2(a/2))
    """
    Phi1 = calc_Phi(1, alpha)
    Phi2 = calc_Phi(2, alpha)
    falpha = H -2.5*np.log10((1-G)*Phi1 + G*Phi2)
    return falpha



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


def check_oe(x0, x1, y0, y1, band):
    """
    Check and return ordinary and extra-ordinary locations
    depending on the fileter.
    """
    # On R-band image, 
    #   Orninary one is located at large x and small y 
    #   Extra-orninary one is located at small x and large y 
    # On B and V-band images,
    #   Orninary one is located at large x and small y 
    #   Extra-orninary one is located at small x and large y 
    if band == "R":
        if x0 > x1 and y0 < y1:
            x_o, y_o = x0, y0
            x_e, y_e = x1, y1
        elif x1 > x0 and y1 < y0:
            x_o, y_o = x1, y1
            x_e, y_e = x0, y0
        else:
            assert False, "Possibly bad detections."

    if band == "B" or band == "V":
        if x0 < x1 and y0 < y1:
            x_o, y_o = x0, y0
            x_e, y_e = x1, y1
        elif x1 < x0 and y1 < y0:
            x_o, y_o = x1, y1
            x_e, y_e = x0, y0
        else:
            assert False, "Possibly bad detections."

    return x_o, y_o, x_e, y_e


def calc_QU(df):
    """
    Calculate stokes Q and U for T60/Dipol-2.

    Parameter
    ---------
    df : pandas.DataFrame
        dataframe with f_o_000, ferr_o_000, f_e_000, ferr_e_000 etc.

    Returns
    -------
    Q, Qerr : float
        Stokes Q normalized by I and its uncertainty
    U, Uerr : float
        Stokes U normalized by I and its uncertainty
    """
    # Check columns
    col = df.columns.tolist()
    assert set(col_Dipol2) < set(col), "Check the input."

    # Based on the method in Namamura Master's thesis
    # Former 8 exposures
    f_o_1, ferr_o_1 = df.at[0, "flux_o_0000"], df.at[0, "fluxerr_o_0000"]
    f_e_1, ferr_e_1 = df.at[0, "flux_e_0000"], df.at[0, "fluxerr_e_0000"]
    f_o_2, ferr_o_2 = df.at[0, "flux_o_0225"], df.at[0, "fluxerr_o_0225"]
    f_e_2, ferr_e_2 = df.at[0, "flux_e_0225"], df.at[0, "fluxerr_e_0225"]
    f_o_3, ferr_o_3 = df.at[0, "flux_o_0450"], df.at[0, "fluxerr_o_0450"]
    f_e_3, ferr_e_3 = df.at[0, "flux_e_0450"], df.at[0, "fluxerr_e_0450"]
    f_o_4, ferr_o_4 = df.at[0, "flux_o_0675"], df.at[0, "fluxerr_o_0675"]
    f_e_4, ferr_e_4 = df.at[0, "flux_e_0675"], df.at[0, "fluxerr_e_0675"]
    f_o_5, ferr_o_5 = df.at[0, "flux_o_0900"], df.at[0, "fluxerr_o_0900"]
    f_e_5, ferr_e_5 = df.at[0, "flux_e_0900"], df.at[0, "fluxerr_e_0900"]
    f_o_6, ferr_o_6 = df.at[0, "flux_o_1125"], df.at[0, "fluxerr_o_1125"]
    f_e_6, ferr_e_6 = df.at[0, "flux_e_1125"], df.at[0, "fluxerr_e_1125"]
    f_o_7, ferr_o_7 = df.at[0, "flux_o_1350"], df.at[0, "fluxerr_o_1350"]
    f_e_7, ferr_e_7 = df.at[0, "flux_e_1350"], df.at[0, "fluxerr_e_1350"]
    f_o_8, ferr_o_8 = df.at[0, "flux_o_1575"], df.at[0, "fluxerr_o_1575"]
    f_e_8, ferr_e_8 = df.at[0, "flux_e_1575"], df.at[0, "fluxerr_e_1575"]

    I1, I2 = f_o_1 + f_e_1, f_o_2 + f_e_2 
    I3, I4 = f_o_3 + f_e_3, f_o_4 + f_e_4 
    I5, I6 = f_o_5 + f_e_5, f_o_6 + f_e_6
    I7, I8 = f_o_7 + f_e_7, f_o_8 + f_e_8

    Q1, U1 = f_o_1 - f_e_1, f_o_2 - f_e_2 
    Q2, U2 = f_o_3 - f_e_3, f_o_4 - f_e_4 
    Q3, U3 = f_o_5 - f_e_5, f_o_6 - f_e_6
    Q4, U4 = f_o_7 - f_e_7, f_o_8 - f_e_8

    # Normalized by I
    Q = (Q1/I1 + Q2/I2 + Q3/I3 + Q4/I4)/4.
    U = (U1/I1 + U2/I2 + U3/I3 + U4/I4)/4.

    # Uncertainties
    I1err, I2err = adderr(ferr_o_1, ferr_e_1), adderr(ferr_o_2, ferr_e_2)
    I3err, I4err = adderr(ferr_o_3, ferr_e_3), adderr(ferr_o_4, ferr_e_4)
    I5err, I6err = adderr(ferr_o_5, ferr_e_5), adderr(ferr_o_6, ferr_e_6)
    I7err, I8err = adderr(ferr_o_7, ferr_e_7), adderr(ferr_o_8, ferr_e_8)

    Q1err, U1err = adderr(ferr_o_1, ferr_e_1), adderr(ferr_o_2, ferr_e_2)
    Q2err, U2err = adderr(ferr_o_3, ferr_e_3), adderr(ferr_o_4, ferr_e_4)
    Q3err, U3err = adderr(ferr_o_5, ferr_e_5), adderr(ferr_o_6, ferr_e_6)
    Q4err, U4err = adderr(ferr_o_7, ferr_e_7), adderr(ferr_o_8, ferr_e_8)
    
    Qerr = adderr(
        diverr(Q1, Q1err, I1, I1err), diverr(Q2, Q2err, I2, I2err), 
        diverr(Q3, Q3err, I3, I3err), diverr(Q4, Q4err, I4, I4err))/4.
    Uerr = adderr(
        diverr(U1, U1err, I1, I1err), diverr(U2, U2err, I2, I2err), 
        diverr(U3, U3err, I3, I3err), diverr(U4, U4err, I4, I4err))/4.

    return Q, Qerr, U, Uerr


# def calc_P(Q, Qerr, U, Uerr):
#     """
#     Calculate linear polarization degree P from stokes Q and U for T60/Dipol-2.
# 
#     Parameters
#     ----------
#     Q, Qerr : float
#         Stokes Q normalized by I and its uncertainty
#     U, Uerr : float
#         Stokes U normalized by I and its uncertainty
# 
#     Returns
#     -------
#     P, Perr : float
#         linear polarization degree and its uncertainty
#     """
#     P = np.sqrt(Q**2 + U**2)
#     Perr = Uerr
#     return P, Perr


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


def time_keeper(func):
    """
    Decorator to measure time.
    """
    # To take over docstring etc.
    @wraps(func)
    def wrapper(*args, **kargs):
        t0 = time.time()
        result = func(*args, **kargs)
        t1 = time.time()
        t_elapse = t1 - t0
        print(f"[time keeper] t_elapse = {t_elapse:.03f} s (func :{func.__name__})")
        return result
    return wrapper


def add_circle_with_radius(ax, x, y, rad, color, ls, label):
    """
    Add circle with radius.
    The circle size is fixed (to 10 by default).

    Parameters
    ----------
    x, y : float
        coordinates of source
    rad : float
        circle radius in pixel
    color : str
        color of circle(s)
    ls : str
        line style of circle(s)
    label : str
        object label in legend
    """
    s_circle = 10.
    ax.scatter(
      x, y, color=color, s=s_circle, lw=1, 
      facecolor="None", alpha=1, label=label)
    ax.add_collection(PatchCollection(
      [Circle((x,y), rad)], color=color, ls=ls, lw=1, facecolor="None", label=None)
      )


def plot_region(
    image, stddev, x_o, y_o, x_e, y_e, radius, out="photregion.png"):
    """
    Plot apertures of circlular photometry.

    Parameters
    ----------
    image : array-like
        object extracted image
    stddev : float
        image background standard deviation
    x_o, y_o : float
        coordinates of ordinary source
    x_e, y_e : float
        coordinates of extra-ordinary source
    radius : float
        aperture radius 
    out : str
        output png filename
    """

    # Plot src image after 5-sigma clipping 
    sigma = 5
    _, vmin, vmax = sigmaclip(image, sigma, sigma)
    ny, nx = image.shape
    fig = plt.figure(figsize=(12,int(12*ny/nx)))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
 
    col_o = "red"
    col_e = "blue"

    # Plot ordinary source
    label = "ordiary"
    add_circle_with_radius(
        ax, x_o, y_o, radius, color=col_o, ls="solid", label=label)
    # Plot extra-ordinary source
    label = "extra-ordiary"
    add_circle_with_radius(
        ax, x_e, y_e, radius, color=col_e, ls="solid", label=label)

    ax.set_xlim([0, nx])
    ax.set_ylim([0, ny])
    ax.legend().get_frame().set_alpha(1.0)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
