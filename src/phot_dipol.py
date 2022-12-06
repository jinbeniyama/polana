#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Do photometry of data obtained with T60 and Dipol-2.
The 1 cycle of input data should be 16.

Example
-------
phot_Dipol2.py (fitslist)
"""
import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser as ap
from datetime import datetime
import sep
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from calcerror import adderr, diverr


ang = np.arange(0, 360, 22.5)
f_o_Dipol2 = [f"flux_o_{int(x*10):04d}" for x in ang]
ferr_o_Dipol2 = [f"fluxerr_o_{int(x*10):04d}" for x in ang]
f_e_Dipol2 = [f"flux_e_{int(x*10):04d}" for x in ang]
ferr_e_Dipol2 = [f"fluxerr_e_{int(x*10):04d}" for x in ang]
col_Dipol2 = f_o_Dipol2 + ferr_o_Dipol2 + f_e_Dipol2 + ferr_e_Dipol2


def remove_background2d_Dipol2(image, mask=None):
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
        diverr(Q1, I1, Q1err, I1err), diverr(Q2, I2, Q2err, I2err), 
        diverr(Q3, I3, Q3err, I3err), diverr(Q4, I4, Q4err, I4err))/4.
    Uerr = adderr(
        diverr(U1, I1, U1err, I1err), diverr(U2, I2, U2err, I2err), 
        diverr(U3, I3, U3err, I3err), diverr(U4, I4, U4err, I4err))/4.

    return Q, Qerr, U, Uerr


def calc_P(Q, Qerr, U, Uerr):
    """
    Calculate linear polarization degree P from stokes Q and U for T60/Dipol-2.

    Parameters
    ----------
    Q, Qerr : float
        Stokes Q normalized by I and its uncertainty
    U, Uerr : float
        Stokes U normalized by I and its uncertainty

    Returns
    -------
    P, Perr : float
        linear polarization degree and its uncertainty
    """
    P = np.sqrt(Q**2 + U**2)
    Perr = Uerr
    return P, Perr


if __name__ == "__main__":
    parser = ap(
        description="Do photometry of data obtained with T60/Dipol-2.")
    parser.add_argument(
        "fits", type=str, nargs="*",
        help="Input fits file")
    parser.add_argument(
        "--fitsdir", type=str, default=".",
        help="Fits directory")
    parser.add_argument(
        "--xr", type=int, nargs=2,
        help="X range in pixel")
    parser.add_argument(
        "--yr", type=int, nargs=2,
        help="Y range in pixel")
    parser.add_argument(
        "--radius", type=float, default=10, 
        help="aperture radius in pixel")
    parser.add_argument(
        "--band", type=str, default="R", 
        help="Filter")
    parser.add_argument(
        "--outdir", type=str, default=".",
        help="Output directory")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output filename")
    args = parser.parse_args()
    

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    fitsdir = args.fitsdir
    radius = args.radius
    band = args.band

 
    # List to save results 
    df_res_list = []

    # Do photometry ===========================================================
    for idx_fi, fi in enumerate(args.fits):
        fi_path = os.path.join(fitsdir, fi)
        src = fits.open(fi_path)[0]
        hdr = src.header 

        # Set inverse gain
        # Read EGAIN in fits header
        # ex) EGAIN   =   1.5199999809265137 /Electronic gain in e-/ADU
        # Based on Nakamura-san's M thesis
        #   R-band gain = 1.48
        #   V-band gain = 1.46
        #   B-band gain = 0
        # Based on JB's check on the data obtained on 2022-11-23
        #   R-band gain = 1.52
        #   V-band gain = 1.48
        #   B-band gain = ? (no EGAIN keyword in the fits)
        if band == "B":
            gain = 0
        else:
            gain = hdr["EGAIN"]

        img = src.data
        ny, nx = img.shape[0], img.shape[1]
        print(f"    Data dimension nx, ny = {nx}, {ny}")

        # Cut image
        if args.xr:
            xmin, xmax = args.xr
            ymin, ymax = args.yr
            img = img[ymin:ymax, xmin:xmax] 
            ny, nx = img.shape[0], img.shape[1]
            print(f"    Data dimension (after cutting) nx, ny = {nx}, {ny}")
            img = img.byteswap().newbyteorder()
            img = img.byteswap().newbyteorder()
        
        # Save photometry info.
        info = dict()
        #info["fits"] = fi
        #info["radius"] = radius

        #gain = src.header[hdr_gain]
        #exp_frame = src.header[hdr_exp]
       
        # Background subtraction ==============================================
        # Convert to float
        img = img.astype(np.float32)
        print(f"Original Mean: {np.mean(img)}")
        print(f"Original Median: {np.median(img)}")
        img, bg_info = remove_background2d_Dipol2(img)
        bgerr = np.round(bg_info["rms"], 2)
        print("")
        print(f"Subtracted Mean: {np.mean(img)}")
        print(f"Subtracted Median: {np.median(img)}")
        print(f"Estimated sky error per pixel is {bgerr} [ADU]")
        print("")
        info["gloabalrms"] = bgerr
        info["level_mean"] = np.mean(img)
        info["level_median"] = np.median(img)
        # Background subtraction ==============================================


        # Source detection ====================================================
        # 3-sigma detection
        dth     = 3
        minarea = 10
        mask    = None
        objects = extract(img, dth, minarea, bgerr, mask)
        N_obj   = len(objects)
        if N_obj != 2:
            print(f"{N_obj} objects detected on {idx_fi}-th fits {fi}.")
            print("Finish the process.")
            #sys.exit()
        
        x0, y0 = objects[0]["x"], objects[0]["y"]
        x1, y1 = objects[1]["x"], objects[1]["y"]
        print(f"Two objects detected at (x, y) = ({x0:.1f}, {y0:.1f}), ({x1:.1f}, {y1:.1f})")
        print("Hopefully ordinary/extra-ordinary sources.")

        x_o, y_o, x_e, y_e = check_oe(x0, x1, y0, y1, band)
        print(f"Ordinary source       at (x, y) = ({x_o:.1f}, {y_o:.1f})")
        print(f"Extra-ordinary source at (x, y) = ({x_e:.1f}, {y_e:.1f})")
        # Source detection ====================================================


        # Check the position angle (ideally 45 deg) =====-=====================
        # Check the position angle (ideally 45 deg) =====-=====================


        # Do photometry =======================================================
        # fluxerr**2 = bgerr_per_pix**2*N_pix + Poission**2
        #            = bgerr_per_pix**2*N_pix + (flux*gain)/gain**2
        flux_o, fluxerr_o, eflag_o = sep.sum_circle(
            img, [x_o], [y_o], r=radius, err=bgerr, gain=gain)
        flux_o, fluxerr_o = float(flux_o), float(fluxerr_o)
        flux_e, fluxerr_e, eflag_e = sep.sum_circle(
            img, [x_e], [y_e], r=radius, err=bgerr, gain=gain)
        flux_e, fluxerr_e = float(flux_e), float(fluxerr_e)
        #assert False, "Photon noise limited? not BG limited?"
        # # Do photometry =======================================================


        # Noise statistics ====================================================
        if idx_fi == 0:
            ## Background noise (in fact sum of background + readout)
            med = np.median(img)
            std = np.std(img)
            #print(f"median, std = {med:.3f}, {std:.3f}")
            #print(f"N_original = {nx*ny}")
            # Clip
            sigma = 3
            for i in range(5):
                img = img[(img < sigma*std) & (img > -sigma*std)]
                std = np.std(img)
                #print(f"N_clipped = {len(img)}")
                #print(f"median, std = {med:.3f}, {std:.3f} (after {i}-th {sigma}-sigma clipping)")
            print(f"Background noise = {std:.3f} ADU/pix")
            print(f", which should be close to bgrms {bgerr:.3f} ADU/pix (returns of sep)\n")
            ## Poission noise 
            # N^2_e = flux_e = flux_ADU*gain(e/ADU)
            # N_ADU = N_e/gain
            # -> N_ADU = sqrt(flux_ADU/gain)
            fluxerr_o_ADU = np.sqrt(flux_o/gain)
            fluxerr_e_ADU = np.sqrt(flux_e/gain)
            print(f"Poisson noise (o, e) = {fluxerr_o_ADU:.2f} {fluxerr_e_ADU:.2f} ADU")
            ## Total noise
            # Aperture area in pix
            N_app = np.pi*radius**2
            # Sum of background noise
            std_sum = np.sqrt(N_app)*std
            sumerr_o = adderr(std_sum, fluxerr_o_ADU)
            sumerr_e = adderr(std_sum, fluxerr_e_ADU)
            print(f"Sum of noise (o, e) = {sumerr_o:.2f} {sumerr_e:.2f} ADU")
            print(f", which should be close to fluxerr {fluxerr_o:.3f} and {fluxerr_e:.3f} ADU (returns of sep)\n")
        # Noise statistics ====================================================

        ang = idx_fi * 22.5 * 10
        info[f"flux_o_{int(ang):04d}"]    = flux_o
        info[f"fluxerr_o_{int(ang):04d}"] = fluxerr_o
        info[f"flux_e_{int(ang):04d}"]    = flux_e
        info[f"fluxerr_e_{int(ang):04d}"] = fluxerr_e
        df_res = pd.DataFrame(info.values(), index=info.keys()).T
        df_res_list.append(df_res)
    df_res = pd.concat(df_res_list, axis=1)
    df_res = df_res.reset_index()
    
    Q, Qerr, U, Uerr = calc_QU(df_res)
    P, Perr = calc_P(Q, Qerr, U, Uerr)
    print(f"  polarization degree P = {P}+-{Perr}")
    
    # Save results
    if args.out:
        out = args.out
    else:
        out = "photres_Dipol2.txt"
    out = os.path.join(outdir, out)
    df_res.to_csv(out, sep=" ", index=False)
    # Do photometry ===========================================================
    assert False, 1


    # Output count in ADU =====================================================
    for idx, row in df_res.iterrows():
        f_o, ferr_o = row["flux_o"], row["fluxerr_o"]
        f_e, ferr_e = row["flux_e"], row["fluxerr_e"]
        print(f" {idx+1}-th fits")
        print(f" Orbinary       Flux = {f_o:7.1f}+-{ferr_o:4.1f}")
        print(f" Extra-orbinary Flux = {f_e:7.1f}+-{ferr_e:4.1f}")
    # Output count in ADU =====================================================

    # Derive Stokes parameters ================================================
    # Derive Stokes parameters ================================================

