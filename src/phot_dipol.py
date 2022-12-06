#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Do photometry for images obtained with T60 and Dipol-2.
The number of input fits for 1 cycle is 16.

Example
-------
phot_dipol.py urat1-473022538_001[7-9]R.fts urat1-473022538_002*R.fts urat1-473022538_003[0-2]R.fts --band R --xr 200 550 --yr 100 350
"""
import os
import sys
import numpy as np
import pandas as pd
from argparse import ArgumentParser as ap
from datetime import datetime
import sep
import astropy.io.fits as fits

from dipolana.util import *


if __name__ == "__main__":
    parser = ap(
        description="Do photometry for images obtained with T60/Dipol-2.")
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
        help="Filter (to set gain)")
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
    print(f"  Aperture radius {radius} pix")
    print(f"  filter {band}-band")

 
    # List to save results 
    df_res_list = []
    for idx_fi, fi in enumerate(args.fits):
        print("")
        print(f"Start analysis of {idx_fi+1}-th fits")
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
        
        # Read 2-d image
        img = src.data
        ny, nx = img.shape[0], img.shape[1]
        print(f"    Data dimension nx, ny = {nx}, {ny}")
        # Exposure time
        t_exp = src.header["EXPTIME"]
        print(f"    Exposure time {t_exp} s")

        # Cut image to remove contaminations
        if args.xr:
            xmin, xmax = args.xr
            ymin, ymax = args.yr
            img = img[ymin:ymax, xmin:xmax] 
            ny, nx = img.shape[0], img.shape[1]
            print(f"    Data dimension (after cutting) nx, ny = {nx}, {ny}")
        
        # Save photometry info.
        info = dict()
       
        # Background subtraction ==============================================
        # Convert to float to suppress error
        img = img.astype(np.float32)
        #print(f"Original Mean: {np.mean(img)}")
        #print(f"Original Median: {np.median(img)}")
        img, bg_info = remove_background2d_Dipol2(img)
        bgerr = np.round(bg_info["rms"], 2)
        #print("")
        #print(f"Subtracted Mean: {np.mean(img)}")
        #print(f"Subtracted Median: {np.median(img)}")
        #print(f"Estimated sky error per pixel is {bgerr} [ADU]")
        #print("")
        info["gloabalrms"] = bgerr
        info["level_mean"] = np.mean(img)
        info["level_median"] = np.median(img)
        # Background subtraction ==============================================


        # Source detection ====================================================
        # 5-sigma detection
        dth     = 5
        minarea = 15
        objects = extract(img, dth, minarea, bgerr, mask=None)
        N_obj   = len(objects)
        # Orbinary + Extra-ordinary 2 sources
        if N_obj != 2:
            print(f"{N_obj} objects detected on {idx_fi}-th fits {fi}.")
            print(f"x: {objects['x']}")
            print(f"y: {objects['y']}")
            print("Finish the process.")
            sys.exit()
        
        x0, y0 = objects[0]["x"], objects[0]["y"]
        x1, y1 = objects[1]["x"], objects[1]["y"]
        print(f"Two objects detected at (x, y) = ({x0:.1f}, {y0:.1f}), ({x1:.1f}, {y1:.1f})")
        print("Hopefully ordinary/extra-ordinary sources.")
        
        # Check (and swap) orbinary and extra-ordinary
        x_o, y_o, x_e, y_e = check_oe(x0, x1, y0, y1, band)
        print(f"Ordinary source       at (x, y) = ({x_o:.1f}, {y_o:.1f})")
        print(f"Extra-ordinary source at (x, y) = ({x_e:.1f}, {y_e:.1f})")

        # Save photometry region
        out = f"photregion_{fi.split('.')[0]}.png"
        out = os.path.join(outdir, out)
        plot_region(img, bgerr, x_o, y_o, x_e, y_e, radius, out)
        # Source detection ====================================================


        # Do photometry =======================================================
        # In ADU, 
        # fluxerr**2 = bgerr_per_pix**2*N_pix + Poission**2
        #            = bgerr_per_pix**2*N_pix + (flux*gain)/gain**2
        flux_o, fluxerr_o, eflag_o = sep.sum_circle(
            img, [x_o], [y_o], r=radius, err=bgerr, gain=gain)
        flux_o, fluxerr_o = float(flux_o), float(fluxerr_o)
        flux_e, fluxerr_e, eflag_e = sep.sum_circle(
            img, [x_e], [y_e], r=radius, err=bgerr, gain=gain)
        flux_e, fluxerr_e = float(flux_e), float(fluxerr_e)
        # Do photometry =====================================================


        # Noise calculation ===================================================
        if idx_fi == 0:
            ## Background noise (sum of background + readout)
            med, std = np.median(img), np.std(img)
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
            # Circular aperture area in pix
            N_app = np.pi*radius**2
            # Sum of background noise
            std_sum = np.sqrt(N_app)*std
            sumerr_o = adderr(std_sum, fluxerr_o_ADU)
            sumerr_e = adderr(std_sum, fluxerr_e_ADU)
            print(f"Sum of noise (o, e) = {sumerr_o:.2f} {sumerr_e:.2f} ADU")
            print(f", which should be close to fluxerr {fluxerr_o:.3f} and {fluxerr_e:.3f} ADU (returns of sep)\n")
        # Noise calculation ===================================================

        # Rotatie angle 22.5 deg (22.5 deg -> 2250)
        ang = idx_fi*22.5*10
        info[f"flux_o_{int(ang):04d}"]    = flux_o
        info[f"fluxerr_o_{int(ang):04d}"] = fluxerr_o
        info[f"flux_e_{int(ang):04d}"]    = flux_e
        info[f"fluxerr_e_{int(ang):04d}"] = fluxerr_e
        df_res = pd.DataFrame(info.values(), index=info.keys()).T
        df_res_list.append(df_res)
    df_res = pd.concat(df_res_list, axis=1)
    df_res = df_res.reset_index()
    
    # Calculate Stokes Q and U
    Q, Qerr, U, Uerr = calc_QU(df_res)
    # Calculate linera polarization degree P
    P, Perr = calc_P(Q, Qerr, U, Uerr)
    
    # Round parameters
    Q, Qerr = round_error(Q, Qerr)
    U, Uerr = round_error(U, Uerr)
    P, Perr = round_error(P, Perr)
    print(f"  Q = {Q}+-{Qerr}")
    print(f"  U = {U}+-{Uerr}")
    print(f"  polarization degree P = {P}+-{Perr}")
    
    # Save results
    if args.out:
        out = args.out
    else:
        out = "photres_Dipol2.txt"
    out = os.path.join(outdir, out)
    df_res.to_csv(out, sep=" ", index=False)


    # # Output flux in ADU =====================================================
    # for idx, row in df_res.iterrows():
    #     f_o, ferr_o = row["flux_o"], row["fluxerr_o"]
    #     f_e, ferr_e = row["flux_e"], row["fluxerr_e"]
    #     print(f" {idx+1}-th fits")
    #     print(f" Orbinary       Flux = {f_o:7.1f}+-{ferr_o:4.1f}")
    #     print(f" Extra-orbinary Flux = {f_e:7.1f}+-{ferr_e:4.1f}")
    # # Output count in ADU =====================================================
