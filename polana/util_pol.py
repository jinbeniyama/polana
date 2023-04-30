#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Useful functions for polarimetry.
"""
import numpy as np
import pandas as pd


def polana_4angle(df, inst):
    """
    Calculate linear polarization degree and position angle 
    with data in four angles of Half-wave plate (0000, 0225, 0450, 0675).
    One important thing is that the definition of q and u are 
    different instrument to instrument!

    Parameter
    ---------
    df : pandas.DataFrame
        input dataframe
    inst : str
        instrument

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
            + (f_e_0000/(f_o_0000*f_o_0450*f_e_0450))*ferr_o_0450**2
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
            + (f_e_0225/(f_o_0225*f_o_0675*f_e_0675))*ferr_o_0675**2
            + ((f_e_0225*f_o_0675)/(f_e_0675**3*f_o_0225))*ferr_e_0675**2
            )
        )

    # Calculate q and u, (q = Q/I and u = U/I)
    # The definisions are sooooooo important.
    # See Kawakami+2021, Akitaya+2014, Geem+2022.
    if inst == "WFGS2" or inst == "HONIR":
        q = (1-Rq)/(1+Rq)
        u = (1-Ru)/(1+Ru)
    # See Ishiguro+2017, Kuroda+2018.
    elif inst == "MSI":
        q = (Rq-1)/(Rq+1)
        u = (Ru-1)/(Ru+1)
    
    # Random errors
    qerr = np.sqrt(
        4./(1+Rq)**4*Rqerr**2
        )
    uerr = np.sqrt(
        4./(1+Ru)**4*Ruerr**2
        )
    
    #   # For test ================================================================
    #   # The consistency was checked by J.B.
    #   A = (
    #       (ferr_e_0000/f_e_0000)**2 + (ferr_o_0000/f_o_0000)**2 
    #       +
    #       (ferr_e_0450/f_e_0450)**2 + (ferr_o_0450/f_o_0450)**2 
    #       )
    #   qerr2 = Rq/(Rq+1)**2 * np.sqrt(A)
    #   B = (
    #       (ferr_e_0225/f_e_0225)**2 + (ferr_o_0225/f_o_0225)**2 
    #       +
    #       (ferr_e_0675/f_e_0675)**2 + (ferr_o_0675/f_o_0675)**2 
    #       )
    #   uerr2 = Ru/(Ru+1)**2 * np.sqrt(B)
    #   print(f"(MSI manual) qerr = {qerr2}")
    #   print(f"(Self calc.) qerr = {qerr}")
    #   print(f"(MSI manual) uerr = {uerr2}")
    #   print(f"(Self calc.) uerr = {uerr}")
    #   # For test ================================================================

    return u, uerr, q, qerr


def cor_poleff(
    df, inst, band, 
    key_q="q", key_u="u", key_qerr_ran="qerr_ran", key_uerr_ran="uerr_ran",
    key_q_cor="q_cor", key_u_cor="u_cor", 
    key_qerr_ran_cor="qerr_ran_cor", key_uerr_ran_cor="uerr_ran_cor", 
    key_qerr_sys_cor="qerr_sys_cor", key_uerr_sys_cor="uerr_sys_cor", 
    key_qerr_cor="qerr_cor", key_uerr_cor="uerr_cor"
    ):
    """
    Do correction about polarization efficiency.

    Parameter
    ---------
    df : pandas.DataFrame
        input dataframe with q, u, and so on.
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
        output dataframe with corrected u, q, and so on.
    """
    if inst == "MSI":
        # from MSI wiki
        if band == "Rc" or "R":
            # 2022-03 
            peff    = 0.9955
            pefferr = 0.0001
        if band == "V":
            # 2022-03 from MSI wiki
            peff    = 0.9959
            pefferr = 0.0002

    if inst == "WFGS2":
        # In Geem+2022b peff=1 (assumption)
        # From Kawakami+2021
        peff    = 0.982
        pefferr = 0.0

    if inst == "HONIR":
        # From Geem+2022b
        # Adopt 0.9758 (sentense) rather than 0.9858 (code)
        peff    = 0.9758
        pefferr = 0.0008

    df[key_q_cor] = df[key_q]/peff
    df[key_u_cor] = df[key_u]/peff 
    
    # Rondom errors
    df[key_qerr_ran_cor] = df[key_qerr_ran]/peff
    df[key_uerr_ran_cor] = df[key_uerr_ran]/peff
    # Systematic error
    df[key_qerr_sys_cor] = np.abs(df[key_q])*pefferr/peff 
    df[key_uerr_sys_cor] = np.abs(df[key_u])*pefferr/peff 
    # Concatenate errors
    df[key_qerr_cor] = np.sqrt(df[key_qerr_ran_cor]**2 + df[key_qerr_sys_cor]**2)
    df[key_uerr_cor] = np.sqrt(df[key_uerr_ran_cor]**2 + df[key_uerr_sys_cor]**2)
    return df


def cor_instpol(
    df, inst, band, 
    key_q="q", key_u="u", 
    key_qerr_ran="qerr_ran", key_uerr_ran="uerr_ran",
    key_qerr_sys="qerr_sys", key_uerr_sys="uerr_sys",
    key_q_cor="q_cor", key_u_cor="u_cor", 
    key_qerr_ran_cor="qerr_ran_cor", key_uerr_ran_cor="uerr_ran_cor", 
    key_qerr_sys_cor="qerr_sys_cor", key_uerr_sys_cor="uerr_sys_cor", 
    key_qerr_cor="qerr_cor", key_uerr_cor="uerr_cor",
    key_insrot1="insrot1", key_insrot2="insrot2"):
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
    key_q, key_qerr_ran, key_qerr_sys : float
        keywords for original q and their errors
    key_q_cor, key_qerr_ran_cor, key_qerr_sys_cor, key_qerr_cor : float
        keywords for corrected q and their errors
    key_insrot1 : str
        keyword for angle of instrument rotator at 0 and 45 deg
    key_insrot2 : str
        keyword for angle of instrument rotator at 22.5 and 67.5 deg

    Return
    ------
    df : pandas.DataFrame
        output dataframe with u, q, etc.
    """

    print("Random errors are not changed in this correction!")

    if inst == "MSI":
        # From MSI wiki
        if band == "Rc" or "R":
            # 2022-03
            qinst    = 0.00862
            qinsterr = 0.00013
            uinst    = 0.00379
            uinsterr = 0.00013
            # 2022-11
            qinst    = 0.00584
            qinsterr = 0.00011
            uinst    = 0.00751
            uinsterr = 0.00011

        elif band == "V":
            # 2022-03
            qinst    = 0.01202
            qinsterr = 0.00013
            uinst    = 0.00530
            uinsterr = 0.00013
            # 2022-11
            qinst    = 0.00785
            qinsterr = 0.00020
            uinst    = 0.01077
            uinsterr = 0.00019

    if inst == "WFGS2":
        # From code in Geem+2022b
        # TODO:Depending on theta_rot?
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
    insrot1    = np.deg2rad(df[key_insrot1])
    insrot1err = 0
    insrot2    = np.deg2rad(df[key_insrot2])
    insrot2err = 0
    print(f"INSROT angle in deg 1, 2 = {insrot1}, {insrot2}")
    
    df[key_q_cor] =  (
        df[key_q] - (np.cos(2*insrot1)*qinst - np.sin(2*insrot1)*uinst)
        )
    df[key_u_cor] =  (
        df[key_u] - (np.sin(2*insrot2)*qinst + np.cos(2*insrot2)*uinst)
        )


    # Rondom errors (not changed)
    df[key_qerr_ran_cor] = df[key_qerr_ran]
    df[key_uerr_ran_cor] = df[key_uerr_ran]
    # Systematic errors
    df[key_qerr_sys_cor] = np.sqrt(
        df[key_qerr_sys]**2
        + (np.cos(2*insrot1)*qinsterr)**2 
        + (np.sin(2*insrot1)*uinsterr)**2 
        )
    df[key_uerr_sys_cor] = np.sqrt(
        df[key_uerr_sys]**2
        + (np.sin(2*insrot2)*qinsterr)**2 
        + (np.cos(2*insrot2)*uinsterr)**2 
        )
    # Concatenate errors
    df[key_qerr_cor] = np.sqrt(df[key_qerr_ran_cor]**2 + df[key_qerr_sys_cor]**2)
    df[key_uerr_cor] = np.sqrt(df[key_uerr_ran_cor]**2 + df[key_uerr_sys_cor]**2)
    return df


def cor_paoffset(
    df, inst, band, 
    key_q="q", key_u="u", 
    key_qerr_ran="qerr_ran", key_uerr_ran="uerr_ran",
    key_qerr_sys="qerr_sys", key_uerr_sys="uerr_sys",
    key_q_cor="q_cor", key_u_cor="u_cor", 
    key_qerr_ran_cor="qerr_ran_cor", key_uerr_ran_cor="uerr_ran_cor", 
    key_qerr_sys_cor="qerr_sys_cor", key_uerr_sys_cor="uerr_sys_cor", 
    key_qerr_cor="qerr_cor", key_uerr_cor="uerr_cor",
    key_instpa="INSTPA"):
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
    key_q, key_qerr_ran, key_qerr_sys : float
        keywords for original q and their errors
    key_q_cor, key_qerr_ran_cor, key_qerr_sys_cor, key_qerr_cor : float
        keywords for corrected q and their errors
    key_instpa : str
        keywords for position angle of instrumet

    Return
    ------
    df : pandas.DataFrame
        output dataframe with u, q, etc.
    """

    # Care should be taken when it comes to theta_off.
    # In Ishiguro+2017, theta_off is 3.38 deg for MSI.
    # The definition of theta_off is 
    #     theta_off = theta_obs - theta_lt.
    # Then theta_off = 3.38.
    # FYI, Jooyeon uses the same definition.

    # In Kawakami+2021, theta_off is -5.19 for WFGS2.
    # The definition of theta_off is 
    #     theta_off = theta_lt - theta_obs.
    # Then theta_off = -5.19.

    # The definition for HONIR is the same with WFGS2. 
    # (Based on JB's mesurements. Thanks to large theta_off, 
    # it is easily tested changing the sign of theta_off in this code below.)

    # We use the same definition of theta_off with Kawakami+2021.
    # Thus we use theta_off = -3.38 (or other negative values) for MSI.
    

    # From MSI wiki
    if inst == "MSI":
        # Here we use theta_off = ''theta_lt - theta_obs'' = -XX (negative).
        if band == "Rc" or "R":
            # 2015-05
            theta_off    = -3.38
            theta_offerr = 0.37
            # 2022-03
            theta_off    = -3.54
            theta_offerr = 0.11
            # 2022-11
            theta_off    = -17.09
            theta_offerr = 0.52
        if band == "V":
            # 2015-05
            theta_off    = -3.82
            thta_offerr = 0.38
            # 2022-03
            theta_off    = -3.84
            thtea_offerr = 0.14
            # 2022-11
            theta_off    = -20.61
            theta_offerr = 0.26

    if inst == "WFGS2":
        if band == "Rc" or "R":
            # From Kawakami+2021. (beta = -5.68 deg in Rc)
            #paoffset    = -5.68
            #paoffseterr =  0.00
            # From Geem+2022b. 
            theta_off    = -5.19
            theta_offerr =  0.00

    if inst == "HONIR":
        if band == "Rc" or "R":
            # From Geem+2022b
            # Seems very good for HD19820 data taken on 2022-12-27 !!
            theta_off    = 36.8
            theta_offerr = 0.13
    

    # TODO: Why this INSTPA is needed.
    #       I think if INSTPA is fixed and always the same.
    #       Thus the INSTPA is cancelled out...
    #       But both Ishiguro+2017 and MSI manual calculate 
    #       theta_rot with theta_off and INSTPA...

    # For MSI,   instpa (df[key_instpa]) = -0.52 (fixed, not zero, 2022-12)
    # For WFGS2, instpa (df[key_instpa]) = 0.0 (fixed, 2022-12)
    # For HONIR, instpa                  = 0.0 (fixed, 2022-12)
    # The sign is sooooooo important.
    
    # In ishiguro+2017 (I17) and MSI manual,
    # thetarot_I17 = theta_off_I17 - INSTPA (i.e., -0.52).
    # Here, the defitition of theta_off is different.
    # (i.e., theta_off_here = -theta_off_I17)
    # The direction of the rotation is inverse.
    # Thus, 
    # theta_rot_here = -theta_rot_I17
    #                = -(theta_off_I17 - INSTPA)
    #                = theta_off_here + INSTPA
    thetarot = theta_off + df[key_instpa]
    instpaerr = 0
    thetaroterr = theta_offerr

    # In radian
    thetarot    = np.deg2rad(thetarot)
    thetaroterr = np.deg2rad(thetaroterr)
    
    # The signs are different from those in Ishiguro+2017.
    # This is due to the difference of the direction of the rotation.
    df[key_q_cor] =  (
        np.cos(2*thetarot)*df[key_q] - np.sin(2*thetarot)*df[key_u]
        )
    df[key_u_cor] =  (
        np.sin(2*thetarot)*df[key_q] + np.cos(2*thetarot)*df[key_u]
        )

    # Rondom errors
    df[key_qerr_ran_cor] = np.sqrt(
        (np.cos(2*thetarot)*df[key_qerr_ran])**2 + (np.sin(2*thetarot)*df[key_uerr_ran])**2
        )
    df[key_uerr_ran_cor] = np.sqrt(
        (np.sin(2*thetarot)*df[key_qerr_ran])**2 + (np.cos(2*thetarot)*df[key_uerr_ran])**2
        )
    # Systematic errors
    df[key_qerr_sys_cor] = np.sqrt(
        (np.cos(2*thetarot)*df[key_qerr_sys])**2 
        + (np.sin(2*thetarot)*df[key_uerr_sys])**2
        + (2*df[key_u_cor]*thetaroterr)**2
        )
    df[key_uerr_sys_cor] = np.sqrt(
        (np.sin(2*thetarot)*df[key_qerr_sys])**2 
        + (np.cos(2*thetarot)*df[key_uerr_sys])**2
        + (2*df[key_q_cor]*thetaroterr)**2
        )
    # Concatenate errors
    df[key_qerr_cor] = np.sqrt(df[key_qerr_ran_cor]**2 + df[key_qerr_sys_cor]**2)
    df[key_uerr_cor] = np.sqrt(df[key_uerr_ran_cor]**2 + df[key_uerr_sys_cor]**2)
    return df


def calc_Ptheta(
    df, 
    key_q="q", key_u="u", 
    key_qerr_ran="qerr_ran", key_uerr_ran="uerr_ran",
    key_qerr_sys="qerr_sys", key_uerr_sys="uerr_sys",
    key_P="P", key_theta="theta", 
    key_Perr_ran="Perr_ran", key_thetaerr_ran="thetaerr_ran",
    key_Perr_sys="Perr_sys", key_thetaerr_sys="thetaerr_sys",
    key_Perr="Perr", key_thetaerr="thetaerr"):

    # Calculate initial P_cor
    df[key_P] = np.sqrt(
        df[key_q]**2 + df[key_u]**2
        )
    # Random error
    df[key_Perr_ran] = np.sqrt(
        df[key_q]**2*df[key_qerr_ran]**2 + df[key_u]**2*df[key_uerr_ran]**2
        )/df[key_P]
    # Systematic error
    df[key_Perr_sys] = np.sqrt(
        df[key_q]**2*df[key_qerr_sys]**2 + df[key_u]**2*df[key_uerr_sys]**2
        )/df[key_P]

    # Correct positive bias in the linear polarization degree
    # using equation in Wardle & Kronberg 1974.
    # Another approximation is introduced in Wang+1977.
    for idx_row, row in df.iterrows():
        P        = row[key_P]
        Perr_ran = row[key_Perr_ran]
        if P**2 - Perr_ran**2 < 0:
            df.at[idx_row, key_P] = 0
        else:
            df.at[idx_row, key_P] =  np.sqrt(P**2 - Perr_ran**2)
    # Concatenate
    df[key_Perr] = np.sqrt(
        df[key_Perr_ran]**2 + df[key_Perr_sys]**2)

    # Calculate theta
    # Assume all signs of u are the same
    # TODO: update
    mean_arctan2 = np.mean(np.arctan2(df[key_u], df[key_q]))
    # The domain of definition of theta is 0 < theta < pi.
    if mean_arctan2 > 0:
        df[key_theta] = 0.5*np.arctan2(df[key_u], df[key_q])
    elif mean_arctan2 < 0:
        # Convert the domain of definition "from -pi to 0" to "from 0 to pi"
        df[key_theta] = 0.5*np.arctan2(df[key_u], df[key_q]) + np.pi
    assert 0 < np.mean(df[key_theta]) < np.pi, "Check the code."

    # arctan2(y, x) returns arctan(y/x)
    # By default, domain of definition of arctan2 is from -pi to pi.
    # But 
    #   the domain of definition of position angle theta is from 0 to pi,
    #   and that of arctan(U/Q) is also from 0 to 2 pi.
    # Note for Excel for Microsoft users:
    #   np.arctan2(u, q) corresponds to ATAN2(q; u).

    # Calculate thetaerr
    # 1. standard calculation
    #df[key_thetaerr] = np.sqrt(
    #    0.25/(1+(df[key_u]/df[key_q])**2)**2*((df[key_uerr]/df[key_q])**2 
    #    + (df[key_u]*df[key_qerr]/df[key_q]**2)**2)    
    # 2. useful result 
    df[key_thetaerr_ran] = 0.5*df[key_Perr_ran]/df[key_P]
    df[key_thetaerr_sys] = 0.5*df[key_Perr_sys]/df[key_P]

    # Concatenate
    df[key_thetaerr] = np.sqrt(
        df[key_thetaerr_ran]**2 + df[key_thetaerr_sys]**2)

    # Error when P = 0 (P < Perr). See Naghizadeh-Khouei & Clarke 1993
    df.loc[df[key_P]==0, key_thetaerr_ran] = 0
    df.loc[df[key_P]==0, key_thetaerr_sys] = 0
    df.loc[df[key_P]==0, key_thetaerr] = np.deg2rad(51.96)

    return df


def projectP2scaplane(
    df, key_Pr, key_Prerr, key_thetar, key_thetarerr, 
    key_P, key_Perr, key_theta, key_thetaerr, key_phi):
    """
    Project the linear polarization degree to scattering plane using 
    object-Sun vector.

    Parameters
    ----------
    df : pandas.Dataframe
        input dataframe

    Return
    ------
    df : pandas.DataFrame
        output dataframe
    """
    # TODO:
    # This function is assuming that phi is almost constant during the night.
    phi = np.mean(df[key_phi])
    # The domain of definition of phi is between 0 < phi < 360 deg.
    # The domain of definition of theta is between 0 < theta < 180 deg.


    # 2023-02-19 ==============================================================
    # See 
    # 0 < phi < 90
    if phi < 90:
        pi = df[key_phi] + 90
    # 90 < phi < 180
    elif phi < 180:
        pi = df[key_phi] - 90
    # 180 < phi < 270
    elif phi < 270:
        pi = df[key_phi] - 90
    # 270 < phi < 360
    else:
        pi = df[key_phi] - 270

    # In degree
    theta_deg = np.rad2deg(df[key_theta])
    # Calculate theta_r 
    # theta_r is angle between normal to the scattering plane and position angle of polarization
    # In radian
    df[key_thetar] = df[key_theta] - np.deg2rad(pi)
    # In degree
    thetar_deg = np.rad2deg(df[key_thetar])
    try:
        print(f"(theta, phi, pi) = ({theta_deg:.1f}, {phi:.1f}, {pi:.1f}) -> theta_r = {thetar_deg:.1f}")
    except:
        print(f"(theta, phi, pi) = ({theta_deg}, {phi}, {pi}) -> theta_r = {thetar_deg}")
    
    df[key_Pr] = df[key_P] * np.cos(2*df[key_thetar])
    # Error of position angle is not changed.
    # Most previous studies 
    # (De Luise+2007, Kuroda+2018, 2021, Geem+2022a, Kiselev+2022)
    # set Prerr = Perr.
    df[key_Prerr] = df[key_Perr]
    df[key_thetarerr] = df[key_thetaerr]
    # 2023-02-19 ==============================================================


    # Geem+2022b ==============================================================
    # https://github.com/Geemjy/Geem_etal_MNRAS_2022
    # Phi range in Geem+2022b is between 68.4 to 235.4 deg.
    # Thus, the equations below are consistent with equations above.
    # if phi + 90 < 180:
    #     pi = phi + 90
    # else:
    #     pi = phi - 90
    # 
    # I suspect that they use different source code to derive their results
    # in table 1. 
    # phi = 235.4
    # theta = 69.9
    # pi = phi - 270
    # theta_r = theta - pi
    # Geem+2022b ==============================================================
    return df


def check_oe_dipol2(x0, x1, y0, y1, band):
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


def calc_QU_dipol2(df):
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
