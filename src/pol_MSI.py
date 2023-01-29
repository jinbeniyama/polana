#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Do photometry for images obtained with MSI.
The format of input file is as follows:
xo yo xe ye fits
273 185 239 67 msi221221_805278.fits
273 183 239 65 msi221221_805279.fits
.
"""
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser as ap
from scipy.spatial import KDTree
import sep
import astropy.io.fits as fits

from polana.util import *


if __name__ == "__main__":
    parser = ap(
        description="Do photometry for images obtained with MSI.")
    parser.add_argument(
        "obj", type=str, 
        help="Object name")
    parser.add_argument(
        "inp", type=str, nargs="*",
        help="Input file with certain format")
    parser.add_argument(
        "--loc", type=str, default="Q33",
        help="Observation location (MPC code)")
    parser.add_argument(
        "--mp", action='store_true',
        help='Save phase angle in the output for minor planet')
    parser.add_argument(
        "--fitsdir", type=str, default=".",
        help="Fits directory")
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

    key_texp = "EXPTIME"
    key_date = "DATE-OBS"
    key_ut = "UT-STR"
    # 0, 45, 22.5, 67.5
    key_ang = "RET-ANG2"
    # Inverse gain e/ADU
    key_gain = "GAIN"
    
    
    alpha_list, P_list, Perr_list = [], [], []
    for x in args.inp:
        # Read input files
        df_in = pd.read_csv(x, sep=" ")
        N_fits = len(df_in)
        N_fits_per_set = 4
        N_set = int(N_fits/N_fits_per_set)
         
        for idx_set in range(N_set):
            print("")
            print(f"Start analysis of {idx_set+1}/{N_set}-th set")
            # List to save results 
            df_res_list = []
            for idx_fi in range(N_fits_per_set):
                fi = df_in.at[idx_set*N_fits_per_set+idx_fi, "fits"]
                print("")
                print(f"    Start analysis of {idx_fi+1}-th fits")
                fi_path = os.path.join(fitsdir, fi)
                src = fits.open(fi_path)[0]
                hdr = src.header 

                # TODO:update
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

                if band == "R":
                    gain = 0
                else:
                    gain = hdr["EGAIN"]
                
                # Read 2-d image
                img = src.data
                ny, nx = img.shape[0], img.shape[1]
                print(f"    Data dimension nx, ny = {nx}, {ny}")
                # Exposure time
                t_exp = src.header[key_texp]
                print(f"    Exposure time {t_exp} s")

                
                # Save photometry info.
                info = dict()
               
                # Background subtraction ==============================================
                # Convert to float to suppress error
                img = img.astype(np.float32)
                #print(f"Original Mean: {np.mean(img)}")
                #print(f"Original Median: {np.median(img)}")
                img, bg_info = remove_background2d_pol(img)
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
                print(f"{N_obj} objects detected on {idx_fi+1}-th fits {fi}.")
                
                # Original coordinates
                xo0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "xo"]
                yo0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "yo"]
                xe0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "xe"]
                ye0 = df_in.at[idx_set*N_fits_per_set+idx_fi, "ye"]

                # # Search the most suitable objects
                # x_base, y_base = objects["x"], objects["y"]
                # tree_base = KDTree(list(zip(x_base, y_base)), leafsize=10)
                # # Ordinary
                # res_o = tree_base.query_ball_point((xo0, yo0), radius)
                # # Extra-ordinary
                # res_e = tree_base.query_ball_point((xe0, ye0), radius)

                # # New barycenters
                # xo1, yo1 = res_o
                # xe1, ye1 = res_e
                # assert False, xo1

                xo1, yo1 = xo0, yo0
                xe1, ye1 = xe0, ye0
                # TODO: Calculate gain 
                # Source detection ====================================================


                # Do photometry =======================================================
                # In ADU, 
                # fluxerr**2 = bgerr_per_pix**2*N_pix + Poission**2
                #            = bgerr_per_pix**2*N_pix + (flux*gain)/gain**2
                flux_o, fluxerr_o, eflag_o = sep.sum_circle(
                    img, [xo1], [yo1], r=radius, err=bgerr, gain=gain)
                flux_o, fluxerr_o = float(flux_o), float(fluxerr_o)
                print(f" xo0, yo0 = {xo0}, {yo0}")
                print(f"  flux_o, fluxerr_o, SNR_o = {flux_o:.2f}, {fluxerr_o:.2f}, {flux_o/fluxerr_o:.1f}")
                assert False, 1
                flux_e, fluxerr_e, eflag_e = sep.sum_circle(
                    img, [xe1], [ye1], r=radius, err=bgerr, gain=gain)
                flux_e, fluxerr_e = float(flux_e), float(fluxerr_e)

                print(f"Ratio e/o = {flux_e/flux_o}")
                # Do photometry =====================================================


                # Noise calculation ===================================================
                # if idx_fi == 0:
                #     ## Background noise (sum of background + readout)
                #     med, std = np.median(img), np.std(img)
                #     #print(f"median, std = {med:.3f}, {std:.3f}")
                #     #print(f"N_original = {nx*ny}")
                #     # Clip
                #     sigma = 3
                #     for i in range(5):
                #         img = img[(img < sigma*std) & (img > -sigma*std)]
                #         std = np.std(img)
                #         #print(f"N_clipped = {len(img)}")
                #         #print(f"median, std = {med:.3f}, {std:.3f} (after {i}-th {sigma}-sigma clipping)")
                #     print(f"Background noise = {std:.3f} ADU/pix")
                #     print(f", which should be close to bgrms {bgerr:.3f} ADU/pix (returns of sep)\n")
                #     ## Poission noise 
                #     # N^2_e = flux_e = flux_ADU*gain(e/ADU)
                #     # N_ADU = N_e/gain
                #     # -> N_ADU = sqrt(flux_ADU/gain)
                #     fluxerr_o_ADU = np.sqrt(flux_o/gain)
                #     fluxerr_e_ADU = np.sqrt(flux_e/gain)
                #     print(f"Poisson noise (o, e) = {fluxerr_o_ADU:.2f} {fluxerr_e_ADU:.2f} ADU")
                #     ## Total noise
                #     # Circular aperture area in pix
                #     N_app = np.pi*radius**2
                #     # Sum of background noise
                #     std_sum = np.sqrt(N_app)*std
                #     sumerr_o = adderr(std_sum, fluxerr_o_ADU)
                #     sumerr_e = adderr(std_sum, fluxerr_e_ADU)
                #     print(f"Sum of noise (o, e) = {sumerr_o:.2f} {sumerr_e:.2f} ADU")
                #     print(f", which should be close to fluxerr {fluxerr_o:.3f} and {fluxerr_e:.3f} ADU (returns of sep)\n")
                # # Noise calculation ===================================================

                # Rotatie angle 0, 22.5 deg (22.5 deg -> 2250)
                ang = hdr[key_ang]
                info[f"flux_o"]    = flux_o
                info[f"fluxerr_o"] = fluxerr_o
                info[f"flux_e"]    = flux_e
                info[f"fluxerr_e"] = fluxerr_e
                info["angle"] = f"{int(ang*10):04d}"
                date = hdr[key_date]
                ut = hdr[key_ut]
                info["utc"] = f"{date}T{ut}"
                df_res = pd.DataFrame(info.values(), index=info.keys()).T
                df_res_list.append(df_res)
            # 1 set results (Length = 4)
            df_res = pd.concat(df_res_list, axis=0)
            df_res = df_res.reset_index()
        
            # Calculate linera polarization degree P
            P, Perr = calc_P_4angle(df_res)
            print(f"  polarization degree P = {P:.4f}+-{Perr:.4f}")
            P_list.append(P)
            Perr_list.append(Perr)

            if args.mp:
                # Obtain phase angle with object name
                # Use the first time
                ut = df_res.at[0, "utc"]
                alpha = utc2alpha(args.obj, ut, args.loc)
                alpha_list.append(alpha)

        
        # Round parameters
        # Save results
        if args.out:
            out = args.out
        else:
            out = "polres_MSI.txt"
        out = os.path.join(outdir, out)
        if args.mp:
            df_all = pd.DataFrame(dict(
                alpha=alpha_list, P=P_list, Perr=Perr_list
                ))
        else:
            df_all = pd.DataFrame(dict(
                P=P_list, Perr=Perr_list
            ))
        df_all.to_csv(out, sep=" ", index=False)
